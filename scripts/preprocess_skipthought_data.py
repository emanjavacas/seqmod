
import itertools
import time

from seqmod.misc import text_processor, Dict
from seqmod import utils as u

from train_skipthought import load_sents, load_pairs


def make_dataset_fname(args, split=None):
    fname = 'l{}-n{}-lv{}-ms{}-ml{}'.format(
        args.lower, args.num, args.level, args.max_size, args.max_len)

    if split is not None:
        fname += '={}'.format(split)

    return fname


def chunk_seq(seq, n):
    finished, buf = False, []
    while not finished:
        try:
            buf.append(next(seq))
        except StopIteration:
            finished = True

        for item in itertools.islice(seq, n-1):
            buf.append(item)

        if buf:
            yield buf
            buf = []


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs='+', required=True)
    parser.add_argument('--output', help='prefix for the stored dataset', required=True)
    parser.add_argument('--max_size', type=int, default=100000)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='char')
    # for large datasets
    parser.add_argument('--num_splits', type=int, default=1)
    args = parser.parse_args()

    infix = make_dataset_fname(args)
    processor = text_processor(
        lower=args.lower, num=args.num, level=args.level)
    d = Dict(eos_token=u.EOS, bos_token=u.BOS, unk_token=u.UNK,
             pad_token=u.PAD, max_size=args.max_size, force_unk=True)

    print("Fitting dictionary")
    # iterate through chunks to be able to count the total number of sequences
    start, num_sents = time.time(), 0
    sents = load_sents(*args.path, max_len=args.max_len, processor=processor)
    for chunk in chunk_seq(sents, 1000):
        num_sents += len(chunk)
        d.partial_fit(chunk)
    d.fit()

    u.save_model(d, '{}.{}.dict'.format(args.output, infix))
    print("... took {:.3f} secs".format(time.time() - start))
    print()

    print("Processing dataset")
    start, chunk_size = time.time(), (num_sents // args.num_splits) + 1
    pairs = load_pairs(*args.path, max_len=args.max_len, processor=processor)

    if args.num_splits == 1:
        fname = '{}.{}'.format(args.output, infix)
        p1, p2 = zip(*pairs)
        u.save_model({'p1': list(d.transform(p1)), 'p2': list(d.transform(p2))}, fname)
    else:
        # iterate through chunks of size proportional to the number of desired chunks
        for idx, chunk in enumerate(chunk_seq(pairs, chunk_size)):
            print("Processing chunk num {}/{}".format(idx+1, args.num_splits))
            p1, p2 = zip(*chunk)
            p1, p2 = d.transform(p1), d.transform(p2)
            infix = make_dataset_fname(args, split=idx+1)
            fname = '{}.{}'.format(args.output, infix)
            u.save_model({'p1': list(p1), 'p2': list(p2)}, fname)
            del p1, p2

    print("... took {:.3f} secs".format(time.time() - start))
