
import os
import sys

import random; random.seed(1001)

import torch
try:
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')
    torch.manual_seed(1001)

from torch import nn

from seqmod.modules.lm import LM
from seqmod import utils as u

from seqmod.misc.trainer import CLMTrainer
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.early_stopping import EarlyStopping


def read_meta(path, sep=';', fileid='filepath'):
    import csv
    metadata = {}
    with open(path) as f:
        rows = csv.reader(f)
        header = next(rows)
        for row in rows:
            row = dict(zip(header, row))
            metadata[row[fileid]] = row
    return metadata


def compute_length(l, length_bins):
    length = len(l)
    output = None
    for length_bin in length_bins[::-1]:
        if length > length_bin:
            output = length_bin
            break
    else:
        output = -1
    return output


def load_lines(rootfiles, metapath,
               input_format='txt',
               include_length=True,
               length_bins=[50, 100, 150, 300],
               categories=()):
    metadata = read_meta(metapath)
    for idx, f in enumerate(rootfiles):
        filename = os.path.basename(f)
        if filename in metadata:
            conds = []
            row = metadata[filename]
            for c in categories:
                conds.append(row[c])
            with open(f, 'r') as lines:
                for l in lines:
                    l = l.strip()
                    if not l:
                        continue
                    lconds = [c for c in conds]
                    if include_length:
                        lconds.append(compute_length(l, length_bins))
                    yield l, lconds
        else:
            print("Couldn't find [%s]" % f)


def tensor_from_files(lines, lang_d, conds_d):
    def chars_gen():
        for line, conds in lines:
            conds = [d.index(c) for d, c in zip(conds_d, conds)]
            for char in next(lang_d.transform([line])):
                yield [char] + conds
    return torch.LongTensor(list(chars_gen())).t().contiguous()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--cond_emb_dim', default=20)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    # dataset
    parser.add_argument('--metapath', required=True)
    parser.add_argument('--filebatch', type=int, default=500)
    parser.add_argument('--path', required=True)
    parser.add_argument('--processed', action='store_true')
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='token')
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--dev_split', type=float, default=0.05)
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=5, type=int)
    parser.add_argument('--decay_every', default=1, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    # - check
    parser.add_argument('--seed', default=None)
    parser.add_argument('--decoding_method', default='sample')
    parser.add_argument('--max_seq_len', default=25, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--hooks_per_epoch', default=5, type=int)
    parser.add_argument('--log_checkpoints', action='store_true')
    parser.add_argument('--visdom_server', default='localhost')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--prefix', default='model', type=str)
    args = parser.parse_args()

    filenames = [os.path.join(args.path, f) for f in os.listdir(args.path)]
    lines_gen = lambda: load_lines(filenames, args.metapath)
    print("Processing datasets...")
    lang_d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                  eos_token=u.EOS, bos_token=u.BOS)
    n_conds = len(next(lines_gen())[1])
    conds_d = [Dict(sequential=False, force_unk=False) for _ in range(n_conds)]
    d = [lang_d] + conds_d

    print("Fitting dicts...")
    ls, cs = zip(*lines_gen())
    print("Fitting language Dict...")
    lang_d.fit(ls)
    print("Fitting condition Dicts...")
    for cond_d, c in zip(conds_d, zip(*cs)):
        cond_d.fit([c])

    print(' * vocabulary size. %d' % len(lang_d))
    # conditional structure
    conds = []
    for idx, subd in enumerate(conds_d):
        print(' * condition [%d] with cardinality %d' % (idx, len(subd)))
        conds.append({'varnum': len(subd), 'emb_dim': args.cond_emb_dim})

    print("Transforming dataset...")
    examples = tuple(tensor_from_files(lines_gen(), lang_d, conds_d))
    train, valid, test = BlockDataset.splits_from_data(
        examples, d, args.batch_size, args.bptt, gpu=args.gpu,
        test=args.test_split, dev=args.dev_split)

    print('Building model...')
    model = LM(len(lang_d), args.emb_dim, args.hid_dim,
               num_layers=args.layers, cell=args.cell,
               dropout=args.dropout, tie_weights=args.tie_weights,
               deepout_layers=args.deepout_layers,
               deepout_act=args.deepout_act,
               word_dropout=args.word_dropout,
               target_code=lang_d.get_unk(), conds=conds)
    model.apply(u.make_initializer())

    if args.gpu:
        model.cuda()

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at,
        decay_every=args.decay_every)
    criterion = nn.CrossEntropyLoss()

    # hooks
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(args.early_stopping)

    # loggers
    visdom_logger = VisdomLogger(
        log_checkpoints=args.log_checkpoints, title=args.prefix, env='lm',
        server='http://' + args.visdom_server)
    std_logger = StdLogger()

    trainer = CLMTrainer(
        model, {'train': train, 'valid': valid, 'test': test},
        criterion, optim)
    num_checkpoints = min(
        1, len(train) // (args.checkpoint * args.hooks_per_epoch))
    trainer.add_loggers(std_logger, visdom_logger)
    trainer.train(args.epochs, args.checkpoint, gpu=args.gpu)
