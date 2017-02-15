
from random import choice, randrange

from dataset import Dataset

import utils as u

from torchtext import data


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
def generate_str(min_len, max_len, vocab, reserved=[u.EOS, u.PAD]):
    randlen = randrange(min_len, max_len)
    return ''.join([choice(vocab[:-len(reserved)]) for _ in range(randlen)])


def generate_set(size, vocab, min_len, max_len, sample_fn):
    for _ in range(size):
        yield sample_fn(generate_str(min_len, max_len, vocab))


def prepare_data(data_generator, char2int, batch_size,
                 align_right=False, gpu=False):
    bos, eos, pad = char2int[u.BOS], char2int[u.EOS], char2int[u.PAD]
    data = sorted(list(data_generator), key=lambda x: len(x[0]))
    src, tgt = zip(*data)
    src = [[char2int[x] for x in seq] + [eos] for seq in src]
    tgt = [[bos] + [char2int[x] for x in seq] + [eos] for seq in tgt]
    return Dataset(src, tgt, batch_size, pad, align_right=align_right, gpu=gpu)


class DummyDataset(data.Dataset):
    def __init__(self, fields, vocab, size,
                 min_len=5, max_len=15, sample_fn=reverse, **kwargs):
        examples = []
        generator = generate_set(size, vocab, min_len, max_len, sample_fn)
        for src, tgt in sorted(generator, key=lambda s: len(s[0])):
            examples.append(
                data.Example.fromlist(
                    [list(src), list(tgt)], fields))

        super(DummyDataset, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    @classmethod
    def splits(cls, fields, vocab, size, dev=0.1, test=0.2, **kwargs):
        train = 1 - sum(split for split in [dev, test] if split)
        assert train > 0, "dev and test proportions must be below 1"
        return tuple(cls(fields, vocab, int(size * split), **kwargs)
                     for split in [train, dev, test] if split)


if __name__ == '__main__':
    import argparse
    import string
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    parser.add_argument('--vocab', default=list(string.ascii_letters))
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--max_len', default=20, type=int)
    parser.add_argument('--train_set_len', default=10000)

    args = parser.parse_args()
    sample_fn = reverse

    def export_random_set(filename, generator):
        src_file, trg_file = filename + '.src.txt', filename + '.trg.txt'
        with open(src_file, 'a+') as src, open(trg_file, 'a+') as trg:
            for s, t in generator:
                src.write(' '.join(list(s)) + '\n')
                trg.write(' '.join(list(t)) + '\n')

    train_data = generate_set(args.train_set_len, args.vocab,
                              min_len=args.min_len, max_len=args.max_len,
                              sample_fn=sample_fn)

    test_data = generate_set(args.train_set_len // 10, args.vocab,
                             min_len=args.min_len, max_len=args.max_len,
                             sample_fn=sample_fn)

    export_random_set(args.train_file, train_data)
    export_random_set(args.test_file, test_data)
