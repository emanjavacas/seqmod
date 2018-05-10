
from collections import OrderedDict

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def init_hidden_for(inp, num_dirs, num_layers, hid_dim, cell,
                    h_0=None, add_init_jitter=False):
    """
    General function for initializing RNN hidden states

    Parameters:
    - inp: torch.Tensor(seq_len, batch_size, dim)
    """
    size = (num_dirs * num_layers, inp.size(1), hid_dim)

    # create h_0
    if h_0 is not None:
        h_0 = h_0.repeat(1, inp.size(1), 1)
    else:
        h_0 = torch.zeros(*size, device=inp.device)

    # eventualy add jitter
    if add_init_jitter:
        h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

    if cell.startswith('LSTM'):
        # compute memory cell
        return h_0, torch.zeros_like(h_0)
    else:
        return h_0


def make_length_mask(lengths):
    """
    Compute binary length mask.

    lengths: torch.Tensor(batch, dtype=int) should be on the desired
        output device.

    Returns:
    --------

    mask: torch.ByteTensor(batch x seq_len)
    """
    maxlen, batch = lengths.detach().max(), len(lengths)
    return torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device) \
                .repeat(batch, 1) \
                .lt(lengths.unsqueeze(1))


def make_dropout_mask(inp, p, size):
    """
    Make dropout mask variable
    """
    return torch.zeros(*size, device=inp.device).bernoulli_(1 - p).div_(1 - p)


def variational_dropout(inp, p=0.0, training=False):
    if inp.dim() != 3:
        raise ValueError("variational dropout expects 3D input")

    if not training or p == 0.0:
        return inp

    seq_len, batch, dim = inp.size()

    return torch.zeros(1, batch, dim, device=inp.device) \
                .bernoulli_(1 - p) \
                .div_(1 - p) \
                .expand(seq_len, batch, dim) * inp


def split(inp, breakpoints, include_last=False):
    """
    Split a 2D input tensor on given breakpoints
    """
    if breakpoints[-1] > len(inp):
        raise ValueError("Breakpoint {} out of range {}".format(
            breakpoints[-1], len(inp)))

    if include_last and breakpoints[-1] == len(inp):
        # ignore last breakpoint if it coincides with sequence length
        breakpoints.pop()

    if include_last:
        breakpoints.append(len(inp))

    output, prev = [], 0
    for breakpoint in breakpoints:
        output.append(inp[prev: breakpoint])
        prev = breakpoint

    return output


def pad_sequence(seqs):
    """
    Apply padding to a sequence of 1D or 2D variables (seq_len [x dim]).
    The seq_len dim is padded up to the longest sequence in the input
    according to the first dimension.

    Returns: (num_words x max_seq_len x dim)
    """
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    padded = []
    for seq, length in zip(seqs, lengths):
        if seq.dim() == 1:
            padded_seq = F.pad(seq, (0, max_len - length))
        else:
            padded_seq = F.pad(seq.t(), (0, max_len - length)).t()
        padded.append(padded_seq)

    return torch.stack(padded), lengths


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
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
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
    perm = torch.tensor(perm) if isinstance(perm, list) else perm
    return x.index_select(dim, perm)


def flip(x, dim):
    """
    Flip (reverse) a tensor along a given dimension

    Taken from: https://github.com/pytorch/pytorch/issues/229

    >>> import torch
    >>> x = torch.tensor([[0, 2, 4], [1, 3, 5]])
    >>> flip(x, 0).tolist()
    [[1, 3, 5], [0, 2, 4]]
    >>> flip(x, 1).tolist()
    [[4, 2, 0], [5, 3, 1]]
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    index = torch.arange(x.size(1)-1, -1, -1, dtype=torch.int64, device=x.device)
    x = x.view(x.size(0), x.size(1), -1)[:, index, :]
    return x.view(xsize)


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
    >>> select_cols(x, list(range(10))).tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    """
    if not isinstance(vec, torch.Tensor):
        vec = torch.tensor(vec, device=t.device)

    return t.gather(1, vec.unsqueeze(1)).squeeze(1)


def pack_sort(inp, lengths, batch_first=False):
    """
    Transform input into PaddedSequence sorting batch by length (as required).
    Also return an index variable that unsorts the output back to the original
    order.

    Parameters:
    -----------
    inp: torch.Tensor(seq_len x batch x dim)
    lengths: LongTensor of length ``batch``

    >>> from torch.nn.utils.rnn import pad_packed_sequence as unpack
    >>> inp = torch.tensor([[1, 3], [2, 4], [0, 5]], dtype=torch.float)
    >>> lengths = torch.tensor([2, 3]) # unsorted order
    >>> sorted_inp, unsort = pack_sort(inp, lengths)
    >>> sorted_inp, _ = unpack(sorted_inp)
    >>> sorted_inp[:, unsort].tolist()  # original order
    [[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]]
    >>> sorted_inp.tolist()  # sorted by length
    [[3.0, 1.0], [4.0, 2.0], [5.0, 0.0]]
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)  # no need to use gpu

    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()

    if batch_first:
        inp = pack_padded_sequence(inp[sort], lengths.tolist())
    else:
        inp = pack_padded_sequence(inp[:, sort], lengths.tolist())

    return inp, unsort


def get_last_token(t, lengths):
    """
    Grab last hidden activation of each batch element according to `lengths`

    #                               ^ (1)      ^ (2)      ^ (3)
    >>> t = torch.tensor([[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]]])
    >>> lengths = torch.tensor([3, 2, 1])
    >>> get_last_token(t, lengths).tolist()
    [[3], [3], [3]]
    """
    seq_len, batch, _ = t.size()
    index = torch.arange(0, batch, dtype=torch.int64, device=lengths.device) * seq_len
    index = index + (lengths - 1)
    t = t.transpose(0, 1).contiguous()  # make it batch first
    t = t.view(seq_len * batch, -1)
    t = t.index_select(0, index)
    t = t.view(batch, -1)
    return t


def detach_vars(data):
    """Split variables from the tree to allow for memory efficient computation
    of softmax losses over a whole sequence."""
    for k, v in data.items():
        if v.requires_grad:
            v = v.detach()
            v.requires_grad = True
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
            inputs.append(data[key]), grads.append(var.grad)

    torch.autograd.backward(inputs, grads, retain_graph=True)


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
