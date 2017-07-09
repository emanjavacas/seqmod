
import os
import sys

import random; random.seed(1001)

import torch
try:
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')
    torch.manual_seed(1001)

import torch.nn as nn

import numpy as np
from itertools import chain

from seqmod.modules.lm import LM
from seqmod import utils as u

#from seqmod.misc.trainer import CLMTrainer
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.preprocess import text_processor
from seqmod.misc.early_stopping import EarlyStopping


def read_meta(path):
    import csv
    metadata = {}
    with open(path) as f:
        rows = csv.reader(f)
        header = next(rows)
        for row in rows:
            row = dict(zip(header, row))
            metadata[row['filepath']] = row
            for c in row['categories'].split(','):
                c = c.lower()
                if 'non-fictie' in c:
                    row['fictie'] = False
                elif 'fictie' in c:
                    row['fictie'] = True
                else:
                    row['fictie'] = 'NA'
    return metadata


def load_lines(rootdir, metapath,
               input_format='txt',
               include_length=True,
               length_bins=[50, 100, 150, 300],
               include_author=True,
               authors=(),
               include_fields=True,
               categories=('fictie',)):
    metadata = read_meta(metapath)
    for idx, f in enumerate(os.listdir(rootdir)):
        if idx % 50 == 0:
            print("Processed [%d] files" % idx)
        if os.path.splitext(f)[0] in metadata:
            conds = []
            row = metadata[os.path.splitext(f)[0]]
            if include_author:
                author = row['author:lastname']
                if (not authors) or author in authors:
                    conds.append(author)
                if include_fields:
                    for c in categories:
                        conds.append(row[c])
            with open(os.path.join(rootdir, f), 'r') as lines:
                for l in lines:
                    l = l.strip()
                    if l:
                        lconds = [c for c in conds]
                        if include_length:
                            length = len(l)
                            for length_bin in length_bins[::-1]:
                                if length > length_bin:
                                    lconds.append(length_bin)
                                    break
                            else:
                                lconds.append(-1)
                        yield l, lconds
        else:
            print("Couldn't find [%s]" % f)


def fit_dicts(lines, lang_d, conds_d):
    ls, cs = zip(*lines())
    lang_d.fit(ls)
    for d, c in zip(conds_d, zip(*cs)):
        d.fit([c])


def flatten(l):
    return list(chain.from_iterable(l))


def expand_conditions(lines, lang_d, conds_d):
    # transform lines
    ls, cs = zip(*lines())
    ls = list(lang_d.transform(ls))
    # transform conditions
    conds = zip(*cs)
    conds = [list(d.transform([c]))[0] for d, c in zip(conds_d, conds)]
    # expand conditions to lists matching sents sizes
    conds = [[[c] * len(l) for c, l in zip(cond, ls)] for cond in conds]
    # concat to single vectors
    ls = flatten(ls)
    return tuple([ls] + [flatten(c) for c in conds])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    # dataset
    parser.add_argument('--metapath', required=True)
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

    if args.processed:
        print("Loading preprocessed datasets...")
        raise NotImplementedError
    else:
        print("Processing datasets...")
        lang_d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                      eos_token=u.EOS, bos_token=u.BOS)
        generator = lambda: load_lines(args.path, args.metapath)
        nconds = len(next(generator())[1])
        conds_d = [Dict(sequential=False, force_unk=False)
                   for _ in range(nconds)]
        print("Fitting dicts...")
        fit_dicts(generator, lang_d, conds_d)
        print("Transforming data...")
        examples = expand_conditions(generator, lang_d, conds_d)
        dataset = BlockDataset(
            tuple(examples), tuple([lang_d] + conds_d),
            args.batch_size, args.bptt, fitted=True, gpu=args.gpu)
        dataset.to_disk('dataset')
