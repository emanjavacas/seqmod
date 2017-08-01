
import os
import random; random.seed(1001)
from collections import OrderedDict
from itertools import groupby

import torch
import torch.nn.init as init
from torch.autograd import Variable


BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'
UNK = '<unk>'


# General utils
def load_model(path):
    if path.endswith('pickle') or path.endswith('pkl'):
        import pickle as p
        load_fn = p.load
    elif path.endswith('pt'):
        load_fn = torch.load
    elif path.endswith('npy'):
        import numpy as np
        load_fn = np.load
    else:
        raise ValueError("Unknown file format [%s]" % path)
    with open(path, 'rb') as f:
        return load_fn(f)


def save_model(model, prefix, d=None, mode='torch'):
    if mode == 'torch':
        import torch
        save_fn, ext = torch.save, 'pt'
    elif mode == 'pickle':
        import pickle as p
        save_fn, ext = p.dump, 'pkl'
    elif mode == 'npy':
        import numpy as np
        save_fn, ext = lambda model, f: np.save(f, model), 'npy'
    else:
        raise ValueError("Unknown mode [%s]" % mode)
    with open(prefix + "." + ext, 'wb') as f:
        save_fn(model, f)
    if d is not None:
        with open(prefix + ".dict." + ext, 'wb') as f:
            save_fn(d, f)


def save_checkpoint(path, model, d, args,
                    ppl=None, vals='cell layers hid_dim emb_dim bptt'):
    """
    Save model together with dictionary and training input arguments.
    """
    vals = '-'.join(['{}{{{}}}'.format(val[0], val) for val in vals.split()])
    fname = vals.format(**args)
    if ppl is not None:
        fname += '-{:.3f}'.format(ppl)
    if not os.path.isdir(path):
        os.mkdir(path)
    fname = os.path.join(path, fname)
    save_model({'model': model, 'd': d, 'args': args}, fname)


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


def tile(t, times):
    """
    Repeat a tensor across an added first dimension a number of times
    """
    return t.unsqueeze(0).expand(times, *t.size())


def swap(x, dim, perm):
    """
    swap the entries of a tensor in given dimension according to a given perm
    - dim: int,
        It has to be less than total number of dimensions in x.
    - perm: list or torch.LongTensor,
        It has to be as long as the specified dimension and it can only contain
        unique elements.
    """
    if isinstance(perm, list):
        perm = torch.LongTensor(perm)
    return Variable(x.data.index_select(dim, perm))


def repackage_bidi(h_or_c):
    """
    In a bidirectional RNN output is (output, (h_n, c_n))
    - output: (seq_len x batch x hid_dim * 2)
    - h_n: (num_layers * 2 x batch x hid_dim)
    - c_n: (num_layers * 2 x batch x hid_dim)

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


def select_cols(t, vec):
    """
    Select columns in t according to vec.

    Parameters:
    -----------
    - t: torch.Tensor (m x n)
    - vec: list with indices of length m with the longest integer at most n-1.
    """
    nrows, ncols = t.size()
    rows = torch.LongTensor(list(range(ncols))).repeat(nrows, 1)
    vec = torch.LongTensor(vec).unsqueeze(1).repeat(1, ncols)
    return t[rows == vec]


# Initializers
def default_weight_init(m, init_range=0.05):
    for p in m.parameters():
        p.data.uniform_(-init_range, init_range)


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


def make_initializer(
        linear={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}},
        rnn={'type': 'xavier_uniform', 'args': {'gain': 1.}},
        rnn_bias={'type': 'constant', 'args': {'val': 0.}},
        emb={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}},
        default={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}}):
        """
        Creates an initializer function customizable on a layer basis.
        """
        rnns = (torch.nn.LSTM, torch.nn.GRU,
                torch.nn.LSTMCell, torch.nn.GRUCell)

        def initializer(m):
            if isinstance(m, (rnns)):  # RNNs
                for p_type, ps in groupby(m.parameters(), rnn_param_type):
                    if p_type == 'weight':
                        for p in ps:
                            getattr(init, rnn['type'])(p, **rnn['args'])
                    else:       # bias
                        for p in ps:
                            getattr(init, rnn_bias['type'])(
                                p, **rnn_bias['args'])
            elif isinstance(m, torch.nn.Linear):  # linear
                for param in m.parameters():
                    getattr(init, linear['type'])(param, **linear['args'])
            elif isinstance(m, torch.nn.Embedding):  # embedding
                for param in m.parameters():
                    getattr(init, emb['type'])(param, **emb['args'])
            else:               # default initializer
                for param in m.parameters():
                    getattr(init, 'uniform')(param, a=-0.05, b=0.05)
        return initializer


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
            return ", ".join("{:6.4f}".format(g.data.norm()) for g in var)
        else:
            if isinstance(var, tuple):
                return "{:6.4f}".format(var[0].data.norm())
            return "{:6.4f}".format(var.data.norm())

    log = "{module}: grad input [{grad_input}], grad output [{grad_output}]"
    print(log.format(
        module=module.__class__.__name__,
        grad_input=grad_norm_str(grad_input),
        grad_output=grad_norm_str(grad_output)))


# Training code
def make_xent_criterion(vocab_size, mask_ids=()):
    weight = torch.ones(vocab_size)
    for mask in mask_ids:
        weight[mask] = 0
    return torch.nn.CrossEntropyLoss(weight=weight)


def format_hyp(score, hyp, hyp_num, d, level='word'):
    """
    Transform a hypothesis into a string for visualization purposes

    score: float, normalized probability
    hyp: list of integers
    hyp_num: int, index of the hypothesis
    d: Dict, dictionary used for fitting the vocabulary
    """
    if level not in ('word', 'char'):
        raise ValueError('level must be "word" or "char"')
    sep = ' ' if level == 'word' else ''
    return '\n* [{hyp}] [Score:{score:.3f}]: {sent}'.format(
        hyp=hyp_num,
        score=score/len(hyp),
        sent=sep.join([d.vocab[c] for c in hyp]))


def make_lm_check_hook(d, seed_text, max_seq_len=25, gpu=False,
                       method='sample', temperature=1, width=5,
                       early_stopping=None, validate=True):
    """
    Make a generator hook for a normal language model
    """

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Checking training...")
        if validate:
            loss = trainer.validate_model()
            trainer.log("info", "Valid loss: %g" % loss)
            trainer.log("info", "Registering early stopping loss...")
            if early_stopping is not None:
                early_stopping.add_checkpoint(loss)
        trainer.log("info", "Generating text...")
        scores, hyps = trainer.model.generate(
            d, seed_text=seed_text, max_seq_len=max_seq_len, gpu=gpu,
            method=method, temperature=temperature, width=width)
        hyps = [format_hyp(score, hyp, hyp_num + 1, d)
                for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
        trainer.log("info", '\n***' + ''.join(hyps) + "\n***")

    return hook


def make_clm_hook(d, sampled_conds=None, max_seq_len=200, gpu=False,
                  method='sample', temperature=1, batch_size=10):
    """
    Make a generator hook for a CLM.

    Parameters:
    -----------
    d: list of Dicts, the first one being the Dict for language, the tail
        being Dicts for conditions in the same order as passed to the model
    sampled_conds: list of lists of str, or int, or None,
        if a list, it is supposed to contain lists specifying the conditions
        for each sample [[cond1, cond2], [cond1, cond2], ...]
        if an int, a total of `sampled_conds` samples will be produced with
        random values for the conditions.
        if None, 5 samples will be taken (resulting in 5 combinations of conds)
    """
    lang_d, *conds_d = d

    # sample conditions if needed
    if isinstance(sampled_conds, int) or sampled_conds is None:
        samples, sampled_conds = sampled_conds or 5, []
        for _ in range(samples):
            sample = [d.index(random.sample(d.vocab, 1)[0]) for d in conds_d]
            sampled_conds.append(sample)

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log('info', 'Generating text...')
        for conds in sampled_conds:
            conds_str = ''
            for idx, (cond_d, sampled_c) in enumerate(zip(conds_d, conds)):
                conds_str += (str(cond_d.vocab[sampled_c]) + '; ')
            trainer.log("info", "\n***\nConditions: " + conds_str)
            scores, hyps = trainer.model.generate(
                lang_d, max_seq_len=max_seq_len, gpu=gpu,
                method=method, temperature=temperature,
                batch_size=batch_size, conds=conds)
            hyps = [format_hyp(score, hyp, hyp_num + 1, lang_d)
                    for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
            trainer.log("info", ''.join(hyps) + "\n")
        trainer.log("info", '***\n')

    return hook
