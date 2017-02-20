
import torch


BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'


# pytorch utils
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
    return t.masked_fill_(t.eq(source_idx), target_idx)


def make_initializer(linear_range):
    def init(m):
        if isinstance(m, torch.nn.Linear):
            pass
