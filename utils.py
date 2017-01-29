
from random import choice, randrange
import torch


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


EOS = '<EOS>'
PAD = '<PAD>'


# generator functions
def identity(string):
    return string, string


def reverse(string):
    return string, string[::-1]


def ntuple(string, n=3):
    return string, ''.join([char * n for char in string])


def double(string):
    return ntuple(string, n=2)


def reversedouble(string):
    return string, ''.join([char + char for char in string[::-1]])


def skipchar(string, skip=1):
    splitby = skip + 1
    return string, ''.join([string[i::splitby] for i in range(splitby)])


def generate_str(min_len, max_len, vocab, reserved=[EOS, PAD]):
    randlen = randrange(min_len, max_len)
    return ''.join([choice(vocab[:-len(reserved)]) for _ in range(randlen)])


def generate_set(size, vocab, min_len=1, max_len=15, sample_fn=reverse):
    for _ in range(size):
        yield sample_fn(generate_str(min_len, max_len, vocab))


class Dataset(object):
    def __init__(self, src_data, tgt_data, batch_size, pad, align_right=True):
        self.pad = pad
        self.align_right = align_right
        self.src = sorted(src_data, key=lambda l: len(l))
        if tgt_data:
            self.tgt = sorted(tgt_data, key=lambda l: len(l))
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.batch_size = batch_size
        self.num_batches = len(self.src) // batch_size

    def _batchify(self, batch_data):
        max_length = max(len(x) for x in batch_data)
        out = torch.LongTensor(len(batch_data), max_length).fill_(self.pad)
        for i in range(len(batch_data)):
            seq = torch.Tensor(batch_data[i])
            seq_length = seq.size(0)
            offset = max_length - seq_length if self.align_right else 0
            out[i].narrow(0, offset, seq_length).copy_(seq)
        out = out.t().contiguous()
        return torch.autograd.Variable(out)

    def __getitem__(self, idx):
        assert idx < self.num_batches, "%d > %d" % (idx, self.num_batches)
        batch_from = idx * self.batch_size
        batch_to = (idx+1)*self.batch_size
        src_batch = self._batchify(self.src[batch_from: batch_to])
        if self.tgt:
            tgt_batch = self._batchify(self.tgt[batch_from: batch_to])
        else:
            tgt_batch = None
        return src_batch, tgt_batch

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':
    import argparse
    import string
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('--vocab', default=list(string.ascii_letters))
    parser.add_argument('--min_len', default=5)
    parser.add_argument('--max_len', default=20)
    parser.add_argument('--train_set_len', default=10000)

    args = parser.parse_args()
    sample_fn = reverse

    def export_random_set(filename, generator):
        with open(filename + '.src.txt', 'a+') as src:
            with open(filename + '.tgt.txt', 'a+') as tgt:
                for s, t in generator:
                    src.write(' '.join(list(s)) + '\n')
                    tgt.write(' '.join(list(t)) + '\n')

    train_data = generate_set(args.train_set_len, args.vocab,
                              min_len=args.min_len, max_len=args.max_len,
                              sample_fn=sample_fn)

    test_data = generate_set(args.train_set_len // 10, args.vocab,
                             min_len=args.min_len, max_len=args.max_len,
                             sample_fn=sample_fn)

    export_random_set(args.train_file, train_data)
    export_random_set(args.test_file, test_data)
