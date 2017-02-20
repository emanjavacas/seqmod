
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
                    sample_fn=reverse, gpu=False, **kwargs):
    src, trg = zip(*generate_set(size, vocab, min_len, max_len, sample_fn))
    src, trg = list(map(list, src)), list(map(list, trg))
    src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS)
    src_dict.fit(src, trg)
    dicts = {'src': src_dict, 'trg': src_dict}
    train, dev = Dataset.splits(
        src, trg, dicts, sort_key=lambda pair: len(pair[0]),
        test=None, batchify=True, batch_size=batch_size, gpu=gpu, **kwargs)
    return train, dev


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
