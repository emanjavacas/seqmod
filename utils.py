
from collections import OrderedDict
from itertools import groupby

import numpy as np
import torch


BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'


# General utils
class EarlyStopping(Exception):
    def __init(self, message, data={}):
        super(EarlyStopping, self).__init__(message)
        self.data = data


# Pytorch utils
def merge_states(state_dict1, state_dict2, merge_map):
    """
    Merge 2 state_dicts mapping parameters in state_dict2 to parameters
    in state_dict1 according to a dict mapping parameter names in
    state_dict2 to parameter names in state_dict1.
    """
    state_dict = OrderedDict()
    for p in state_dict2:
        if p in merge_map:
            target_p = merge_map[p]
            assert target_p in state_dict1, \
                "Wrong target param [%s]" % target_p
            state_dict[target_p] = [p]
        else:
            state_dict[target_p] = state_dict1[target_p]
    return state_dict


def tile(t, times):
    """
    Repeat a tensor across an added first dimension a number of times
    """
    return t.unsqueeze(0).expand(times, *t.size())


def bmv(bm, v):
    """
    Parameters:
    -----------
    bm: (batch x dim1 x dim2)
    v: (dim2)

    Returns: batch-wise product of m and v (batch x dim1 x 1)
    """
    batch = bm.size(0)
    # v -> (batch x dim2 x 1)
    bv = v.unsqueeze(0).expand(batch, v.size(0)).unsqueeze(2)
    return bm.bmm(bv)


def repeat(x, size):
    """
    Utility function for the missing (as of 7. Feb. 2017) repeat
    method for Variable's
    """
    return torch.autograd.Variable(x.data.repeat(*size))


def swap(x, dim, perm):
    """
    swap the entries of a tensor in given dimension according to a given perm
    dim: int,
        It has to be less than total number of dimensions in x.
    perm: list or torch.LongTensor,
        It has to be as long as the specified dimension and it can only contain
        unique elements.
    """
    if isinstance(perm, list):
        perm = torch.LongTensor(perm)
    return torch.autograd.Variable(x.data.index_select(dim, perm))


def repackage_bidi(h_or_c):
    """
    In a bidirectional RNN output is (output, (h_n, c_n))
      output: (seq_len x batch x hid_dim * 2)
      h_n: (num_layers * 2 x batch x hid_dim)
      c_n: (num_layers * 2 x batch x hid_dim)

    This function turns a hidden input into:
      (num_layers x batch x hid_dim * 2)
    """
    layers_2, bs, hid_dim = h_or_c.size()
    return h_or_c.view(layers_2 // 2, 2, bs, hid_dim) \
                 .transpose(1, 2).contiguous() \
                 .view(layers_2 // 2, bs, hid_dim * 2)


def unpackage_bidi(h_or_c):
    layers, bs, hid_dim_2 = h_or_c.size()
    return h_or_c.view(layers, bs, 2, hid_dim_2 // 2) \
                 .transpose(1, 2).contiguous() \
                 .view(layers * 2, bs, hid_dim_2 // 2)


def map_index(t, source_idx, target_idx):
    """
    Map a source integer to a target integer across all dims of a LongTensor.
    """
    return t.masked_fill_(t.eq(source_idx), target_idx)


# Initializers
def default_weight_init(m, init_range=0.05):
    for p in m.parameters():
        p.data.uniform_(-init_range, init_range)


def get_fans(weight):
    fan_in = weight.size(1)
    fan_out = weight.size(0)
    return fan_in, fan_out


def rnn_param_type(param):
    """
    Distinguish between bias and state weight params inside an RNN.
    Criterion: biases are vectors, weights are matrices.
    """
    if len(param.size()) == 2:
        return "weight"
    elif len(param.size()) == 1:
        return "bias"
    raise ValueError("Unkown param shape of size [%d]" % param.size())


class Initializer(object):
    """
    Stateless class grouping different initialization functions.
    """
    @staticmethod
    def glorot_uniform(weight, gain=1.0):
        fan_in, fan_out = get_fans(weight)
        a = gain * np.sqrt(2. / (fan_in + fan_out))
        weight.data.uniform_(-a, a)

    @staticmethod
    def glorot_normal(weight, gain=1.0):
        fan_in, fan_out = get_fans(weight)
        std = gain * np.sqrt(2. / (fan_in + fan_out))
        weight.data.normal_(std=std)

    @staticmethod
    def constant(param, value=0.0):
        param.data = param.data.zero_() + value

    @staticmethod
    def uniform(param, min_scale, max_scale):
        assert max_scale >= min_scale, \
            "Wrong scale [%d < %d]" % (max_scale, min_scale)
        param.data.uniform_(min_scale, max_scale)

    @staticmethod
    def orthogonal(param, gain=1.0):
        normal = np.random.standard_normal(size=param.size())
        u, _, v = np.linalg.svd(normal, full_matrices=False)
        if u.shape == param.data.numpy().shape:
            param.data.copy_(torch.from_numpy(u))
        else:
            param.data.copy_(torch.from_numpy(v))
        param.data.mul_(gain)

    @classmethod
    def make_initializer(
            cls,
            linear={'type': 'uniform',
                    'args': {'min_scale': -0.05, 'max_scale': 0.05}},
            rnn={'type': 'glorot_uniform', 'args': {'gain': 1.}},
            rnn_bias={'type': 'constant', 'args': {'value': 0.}},
            emb={'type': 'uniform',
                 'args': {'min_scale': -0.05, 'max_scale': 0.05}},
            default={'type': 'uniform',
                     'args': {'min_scale': -0.05, 'max_scale': 0.05}}):
        """
        Creates an initializer function customizable on a layer basis.
        """
        rnns = (
            torch.nn.LSTM, torch.nn.GRU, torch.nn.LSTMCell, torch.nn.GRUCell)

        def init(m):
            if isinstance(m, (rnns)):  # RNNs
                for p_type, ps in groupby(m.parameters(), rnn_param_type):
                    if p_type == 'weight':
                        for p in ps:
                            getattr(cls, rnn['type'])(p, **rnn['args'])
                    else:       # bias
                        for p in ps:
                            getattr(cls, rnn_bias['type'])(
                                p, **rnn_bias['args'])
            elif isinstance(m, torch.nn.Linear):  # LINEAR
                for param in m.parameters():
                    getattr(cls, linear['type'])(param, **linear['args'])
            elif isinstance(m, torch.nn.Embedding):  # EMBEDDING
                for param in m.parameters():
                    getattr(cls, emb['type'])(param, **emb['args'])
            else:               # default initializer
                for param in m.parameters():
                    cls.uniform(param, -0.05, 0.05)
        return init


# Hooks
def has_trainable_parameters(module):
    """
    Determines whether a given module has any trainable parameters at all
    as per parameter.requires_grad
    """
    num_trainable_params = sum(p.requires_grad for p in module.parameters())
    return num_trainable_params > 0


def log_grad(module, grad_input, grad_output):
    """
    Logs input and output gradients for a given module.
    To be used by module.register_backward_hook or module.register_forward_hook
    """
    if not has_trainable_parameters(module):
        return

    def grad_norm_str(var):
        if isinstance(var, tuple) and len(var) > 1:
            return ", ".join("%6.4f" % g.data.norm() for g in var)
        else:
            if isinstance(var, tuple):
                return "%6.4f" % var[0].data.norm()
            return "%6.4f" % var.data.norm()

    log = "{module}: grad input [{grad_input}], grad output [{grad_output}]"
    print(log.format(
        module=module.__class__.__name__,
        grad_input=grad_norm_str(grad_input),
        grad_output=grad_norm_str(grad_output)))
