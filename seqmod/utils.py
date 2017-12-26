
import os
import yaml
from datetime import datetime
import random; random.seed(1001)
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.init as init
from torch.autograd import Variable

from seqmod.modules.rnn import (
    StackedLSTM, StackedGRU,
    StackedNormalizedGRU, NormalizedGRUCell, NormalizedGRU,
    RHNCoupled, RHN)


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
        raise ValueError("Unknown file format [{}]".format(path))
    with open(path, 'rb') as f:
        return load_fn(f)


def save_model(model, prefix, d=None, mode='torch'):
    """
    Save model using a preferred method. Model gets saved to `prefix.ext`,
    where `ext` is derived from the selected mode. Pass `d` if you want to
    also save a corresponding dictionary to `prefix.dict.ext`.

    If target directory path doesn't exist, it will fail.
    """
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
        raise ValueError("Unknown mode [{}]".format(mode))

    with open(prefix + "." + ext, 'wb') as f:
        save_fn(model, f)

    if d is not None:
        with open(prefix + ".dict." + ext, 'wb') as f:
            save_fn(d, f)


def save_checkpoint(parent, model, d, args, ppl=None, suffix=None):
    """
    Save model together with dictionary and training input arguments.
    If target parent path doesn't exist it will be created.
    """
    dirpath = model.__class__.__name__
    dirpath += '-{}'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    if ppl is not None:
        dirpath += '-{:.3f}'.format(ppl)
    if suffix is not None:
        dirpath += '-{}'.format(suffix)
    dirpath = os.path.join(parent, dirpath)

    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    # save model with dictionary
    save_model(model, os.path.join(dirpath, 'model'), d=d)

    # save hyperparameters
    with open(os.path.join(dirpath, 'params.yml'), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)

    return dirpath


class EmbeddingLoader(object):

    MODES = ('glove', 'fasttext')

    def __init__(self, filepath, mode):
        if not os.path.isfile(filepath):
            raise ValueError("Couldn't find file {}".format(filepath))

        if mode.lower() not in EmbeddingLoader.MODES:
            raise ValueError("Unknown file mode {}".format(mode))

        self.filepath = filepath
        self.mode = mode.lower()

        self.has_header = False
        if self.mode == 'fasttext':
            self.has_header = True

    def reader(self):
        with open(self.filepath, 'r') as f:

            if self.has_header:
                next(f)

            for line in f:
                w, *vec = line.split()

                yield w, vec

    def load(self, words=None):
        vectors, outwords = [], []

        if words is not None:
            words = set(words)

        for word, vec in self.reader():
            if words is not None and word not in words:
                continue

            vectors.append(list(map(float, vec)))
            outwords.append(word)

        return vectors, outwords


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
                "Wrong target param [{}]".format(target_p)
            state_dict[target_p] = [p]
        else:
            state_dict[target_p] = state_dict1[target_p]
    return state_dict


def make_length_mask(lengths):
    """
    Compute binary length mask.

    lengths: Variable torch.LongTensor(batch) should be on the desired
        output device.

    Returns:
    --------

    mask: torch.ByteTensor(batch x seq_len)
    """
    maxlen, batch = lengths.data.max(), len(lengths)
    mask = torch.arange(0, maxlen, out=lengths.data.new()) \
                .repeat(batch, 1) \
                .lt(lengths.data.unsqueeze(1))
    return Variable(mask, volatile=lengths.volatile)


def repackage_bidi(h_or_c):
    """
    In a bidirectional RNN output is (output, (h_n, c_n))
    - output: (seq_len x batch x hid_dim * 2)
    - h_n: (num_layers * 2 x batch x hid_dim)
    - c_n: (num_layers * 2 x batch x hid_dim)

    This function turns a hidden output into:
        (num_layers x batch x hid_dim * 2)
    """
    layers_2, bs, hid_dim = h_or_c.size()
    return h_or_c.view(layers_2 // 2, 2, bs, hid_dim) \
                 .transpose(1, 2).contiguous() \
                 .view(layers_2 // 2, bs, hid_dim * 2)


def repackage_hidden(h):
    """
    Detach hidden from previous graph
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def swap(x, dim, perm):
    """
    Swap the entries of a tensor in given dimension according to a given perm

    Parameters:
    -----------

    - dim: int, less than total number of dimensions in x.
    - perm: list or torch.LongTensor such that the the perm[i] entry in the
        selected dimension of swap(x) contains the x[i] entry of the original
        tensor.
    """
    if isinstance(perm, list):
        perm = torch.LongTensor(perm)
    return x.index_select(dim, torch.autograd.Variable(perm))


def map_index(t, source_idx, target_idx):
    """
    Map a source integer to a target integer across all dims of a LongTensor.
    """
    return t.masked_fill_(t.eq(source_idx), target_idx)


def select_cols(t, vec):
    """
    `vec[i]` contains the index of the column to pick from the ith row  in `t`

    Parameters
    ----------

    - t: torch.Tensor (m x n)
    - vec: list or torch.LongTensor of size equal to number of rows in t

    >>> x = torch.arange(0, 10).repeat(10, 1).t()  # [[0, ...], [1, ...], ...]
    >>> list(select_cols(x, list(range(10))))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    """
    if isinstance(vec, list):
        vec = torch.LongTensor(vec)
    return t.gather(1, vec.unsqueeze(1)).squeeze(1)


def _wrap_variable(t, volatile, gpu):
    if len(t) == 0:             # don't wrap empty vectors
        return t
    if isinstance(t, np.ndarray):
        if t.dtype == np.int64 or t.dtype == np.int32:
            t = torch.LongTensor(t)
        else:
            t = torch.Tensor(t)
    elif isinstance(t, list):
        if isinstance(t[0], int):
            t = torch.LongTensor(t)
        else:
            t = torch.Tensor(t)
    t = Variable(t, volatile=volatile)
    if gpu:
        return t.cuda()
    return t


def wrap_variables(tensor, volatile=False, gpu=False):
    "Transform tensors into variables"
    if isinstance(tensor, tuple):
        return tuple(wrap_variables(t, volatile, gpu) for t in tensor)
    return _wrap_variable(tensor, volatile, gpu)


def unwrap_variables(variables):
    "Transform variables into tensors"
    return [v.data[0] if isinstance(v, Variable) else v for v in variables]


def detach_vars(data):
    """Split variables from the tree to allow for memory efficient computation
    of softmax losses over a whole sequence."""
    for k, v in data.items():
        if v.requires_grad:
            v = Variable(v.data, requires_grad=True, volatile=False)
        yield k, v


def shards(data, size=25, test=False):
    """
    Generator over variables that will be involved in a costly loss computation
    such as the softmax. It yields dictionaries of the same form as the input,
    where the variables have been splitted in smaller shards and detach from
    the graph. It expects the consumer to back propagate through them in shards
    of given a size. After all shards are consumed, the generator will take
    care of backprop further from the input using the accumulated gradients.
    """
    # Inspired by www.github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Loss.py
    if test:
        yield data
        return

    detached = dict(detach_vars(data))
    splits = ((key, torch.split(v, size)) for key, v in detached.items())
    keys, splits = zip(*splits)

    for split in zip(*splits):
        yield dict(zip(keys, split))  # go and accumulate some loss

    inputs, grads = [], []
    for key, var in detached.items():
        if var.grad is not None:
            inputs.append(data[key]), grads.append(var.grad.data)

    torch.autograd.backward(inputs, grads, retain_graph=True)


# Initializers
def is_bias(param_name):
    return 'bias' in param_name


def make_initializer(
        linear={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}},
        linear_bias={'type': 'constant', 'args': {'val': 0.}},
        rnn={'type': 'xavier_uniform', 'args': {'gain': 1.}},
        rnn_bias={'type': 'constant', 'args': {'val': 0.}},
        cnn_bias={'type': 'constant', 'args': {'val': 0.}},
        emb={'type': 'normal', 'args': {'mean': 0, 'std': 1}},
        default={'type': 'uniform', 'args': {'a': -0.05, 'b': 0.05}}):

    rnns = (torch.nn.LSTM, torch.nn.GRU,
            torch.nn.LSTMCell, torch.nn.GRUCell,
            StackedGRU, StackedLSTM, NormalizedGRU,
            NormalizedGRUCell, StackedNormalizedGRU)

    convs = (torch.nn.Conv1d, torch.nn.Conv2d)

    def initializer(m):

        if isinstance(m, (rnns)):  # RNNs
            for p_name, p in m.named_parameters():
                if hasattr(p, 'custom'):
                    continue
                if is_bias(p_name):
                    getattr(init, rnn_bias['type'])(p, **rnn_bias['args'])
                else:
                    getattr(init, rnn['type'])(p, **rnn['args'])

        elif isinstance(m, torch.nn.Linear):  # linear
            for p_name, p in m.named_parameters():
                if hasattr(p, 'custom'):
                    continue
                if is_bias(p_name):
                    getattr(init, linear_bias['type'])(p, **linear_bias['args'])
                else:
                    getattr(init, linear['type'])(p, **linear['args'])

        elif isinstance(m, torch.nn.Embedding):  # embedding
            for p in m.parameters():
                if hasattr(p, 'custom'):
                    continue
                getattr(init, emb['type'])(p, **emb['args'])

        elif isinstance(m, convs):
            for p_name, p in m.named_parameters():
                if hasattr(p, 'custom'):
                    continue
                if is_bias(p_name):
                    getattr(init, cnn_bias['type'])(p, **cnn_bias['args'])
                else:
                    # Karpathy: http://cs231n.github.io/neural-networks-2/#init
                    # -> scale weight vector by square root of its fan-in...
                    # fan_in, _ = init._calculate_fan_in_and_fan_out(p)
                    # init.normal(p, mean=0, std=math.sqrt(fan_in))
                    init.xavier_normal(p)

    return initializer


def initialize_model(model, overwrite_custom=True, **init_ops):
    """
    Applies initializer function, eventually calling any module
    specific custom initializers. Modules can implement custom initialization
    methods `custom_init` to overwrite the general initialization.
    Additionally, parameters can have an additional custom attribute
    set to True and `initialize_model` won't touch them.

    Parameters
    ----------

    model: nn.Module to be initialize
    overwrite_custom: bool, whether to use submodule's custom_init
        to overwrite user input initializer values.
    init_ops: any opts passed to make_initializer
    """
    model.apply(make_initializer(**init_ops))

    if overwrite_custom:
        for m in model.modules():  # overwrite custom
            if hasattr(m, 'custom_init'):
                m.custom_init()


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


def make_lm_hook(d, seed_texts=None, max_seq_len=25, gpu=False,
                 method='sample', temperature=1, width=5,
                 early_stopping=None, validate=True):
    """
    Make a generator hook for a normal language model
    """

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Checking training...")
        if validate:
            loss = sum(trainer.validate_model().pack())
            trainer.log("info", "Valid loss: {:g}".format(loss))
            trainer.log("info", "Registering early stopping loss...")
            if early_stopping is not None:
                early_stopping.add_checkpoint(loss)
        trainer.log("info", "Generating text...")
        scores, hyps = trainer.model.generate(
            d, seed_texts=seed_texts, max_seq_len=max_seq_len, gpu=gpu,
            method=method, temperature=temperature, width=width)
        hyps = [format_hyp(score, hyp, hyp_num + 1, d)
                for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
        trainer.log("info", '\n***' + ''.join(hyps) + "\n***")

    return hook


def make_mlm_hook(d, seed_texts=None, max_seq_len=25, gpu=False,
                  method='sample', temperature=1, width=5,
                  early_stopping=None, validate=True):
    """
    Make a generator hook for a normal language model
    """

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Checking training...")
        if validate:
            loss = sum(trainer.validate_model().pack())
            trainer.log("info", "Valid loss: {:g}".format(loss))
            trainer.log("info", "Registering early stopping loss...")
            if early_stopping is not None:
                early_stopping.add_checkpoint(loss)
        trainer.log("info", "Generating text...")
        for head in trainer.model.project:
            trainer.log("info", "Head: {}".format(head))
            scores, hyps = trainer.model.generate(
                d, head=head, seed_texts=seed_texts, max_seq_len=max_seq_len,
                gpu=gpu, method=method, temperature=temperature, width=width)
            hyps = [format_hyp(score, hyp, hyp_num + 1, d)
                    for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
            trainer.log("info", '\n***' + ''.join(hyps) + "\n***")

    return hook


def make_clm_hook(d, sampled_conds=None, max_seq_len=200, gpu=False,
                  method='sample', temperature=1, batch_size=10):
    """
    Make a generator hook for a CLM.

    Parameters
    ----------

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


def make_schedule_hook(scheduler, verbose=True):

    def hook(trainer, epoch, batch, checkpoint):
        batches = len(trainer.datasets['train'])
        old_rate = trainer.model.exposure_rate
        new_rate = scheduler(epoch * batches + batch)
        trainer.model.exposure_rate = new_rate

        if verbose:
            tmpl = "Exposure rate: [{:.3f}] -> [{:.3f}]"
            trainer.log("info", tmpl.format(old_rate, new_rate))

    return hook
