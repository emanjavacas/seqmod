
from random import choice, randrange
from dataset import Dataset, Dict
import utils as u


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


def load_dummy_data(size, vocab, batch_size, min_len=5, max_len=15,
                    sample_fn=reverse, gpu=False, split=True, **kwargs):
    src, trg = zip(*generate_set(size, vocab, min_len, max_len, sample_fn))
    src, trg = list(map(list, src)), list(map(list, trg))
    src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS)
    src_dict.fit(src, trg)
    dicts = {'src': src_dict, 'trg': src_dict}
    dataset = Dataset(src, trg, dicts)
    if split:
        train, dev = dataset.splits(
            sort_key=lambda pair: len(pair[0]), test=None, batchify=True,
            batch_size=batch_size, gpu=gpu, **kwargs)
        return train, dev, src_dict
    else:
        return dataset


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

    dataset = load_dummy_data(
        args.train_len, args.vocab, None,
        min_len=args.min_len, max_len=args.max_len, split=False)

    with open(args.path, 'wb') as f:
        dataset.to_disk(f)
