
from random import choice, randrange

from misc.dataset import PairedDataset, Dict

from modules import utils as u


# string generator functions
def identity(string):
    return string, string


def reverse(string):
    return string, string[::-1]


def ntuple(string, n=1):
    return string, ''.join([char * n for char in string])


def double(string):
    return ntuple(string, n=2)


def triple(string):
    return ntuple(string, n=3)


def quadruple(string):
    return ntuple(string, n=4)


def reversedouble(string):
    return string, ''.join([char + char for char in string[::-1]])


def skipchar(string, skip=1):
    splitby = skip + 1
    return string, ''.join([string[i::splitby] for i in range(splitby)])


def skip2(string):
    return skipchar(string, skip=2)


def skip3(string):
    return skipchar(string, skip=3)


def skip1reverse(string):
    skip = skipchar(string, skip=1)[1]
    return reverse(skip)


def skip2reverse(string):
    skip = skipchar(string, skip=2)[1]
    return reverse(skip)


# dataset
def generate_str(min_len, max_len, vocab):
    randlen = randrange(min_len, max_len)
    return ''.join([choice(vocab) for _ in range(randlen)])


def generate_set(size, vocab, min_len, max_len, sample_fn):
    for _ in range(size):
        yield sample_fn(generate_str(min_len, max_len, vocab))


if __name__ == '__main__':
    import argparse
    import string
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--vocab', default=list(string.ascii_letters))
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=20, type=int)
    parser.add_argument('--train_len', default=10000, type=int)

    args = parser.parse_args()
    sample_fn = reverse

    src, trg = zip(*generate_set(
        args.train_len,
        args.vocab,
        args.min_len,
        args.max_len,
        sample_fn))
    src, trg = list(map(list, src)), list(map(list, trg))
    src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS)
    src_dict.fit(src, trg)
    dataset = PairedDataset(
        src, trg, {'src': src_dict, 'trg': src_dict}
    ).to_disk(args.path)
