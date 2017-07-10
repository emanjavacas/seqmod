
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
from itertools import chain, islice

from seqmod.modules.lm import LM
from seqmod import utils as u

from seqmod.misc.trainer import LMTrainer, repackage_hidden
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.preprocess import text_processor
from seqmod.misc.early_stopping import EarlyStopping


class CLMTrainer(LMTrainer):
    def run_batch(self, batch_data, dataset='train', **kwargs):
        (src, *conds), (trg, *_) = batch_data
        hidden = self.batch_state.get('hidden', None)
        output, hidden, _ = self.model(src, hidden=hidden, conds=conds)
        self.batch_state['hidden'] = repackage_hidden(hidden)
        loss = self.criterion(output, trg.view(-1))
        if dataset == 'train':
            loss.backward(), self.optimizer_step()
        return (loss.data[0], )

    def on_batch_end(self, batch, loss):
        if hasattr(self, 'reset_hidden'):
            self.batch_state['hidden'].zero_()

    def num_batch_examples(self, batch_data):
        (src, *_), _ = batch_data
        return src.nelement()


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


def load_lines(rootfiles, metapath,
               input_format='txt',
               include_length=True,
               length_bins=[50, 100, 150, 300],
               include_author=True,
               authors=(),
               include_fields=True,
               categories=('fictie',)):
    metadata = read_meta(metapath)
    for idx, f in enumerate(rootfiles):
        if idx % 50 == 0:
            print("Processed [%d] files" % idx)
        filename = os.path.basename(f)
        if os.path.splitext(filename)[0] in metadata:
            conds = []
            row = metadata[os.path.splitext(filename)[0]]
            if include_author:
                author = row['author:lastname']
                if (not authors) or author in authors:
                    conds.append(author)
                if include_fields:
                    for c in categories:
                        conds.append(row[c])
            with open(f, 'r') as lines:
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


def flatten(l):
    return list(chain.from_iterable(l))


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def expand_conditions(lines, d):
    lang_d, *conds_d = d
    ls, cs = zip(*lines)
    # transform lines
    ls = list(lang_d.transform(ls))
    # transform conditions
    conds = zip(*cs)
    conds = [list(d.transform([c]))[0] for d, c in zip(conds_d, conds)]
    # expand conditions to lists matching sents sizes
    conds = [[[c] * len(l) for c, l in zip(cond, ls)] for cond in conds]
    # concat to single vectors
    ls = flatten(ls)
    return tuple([ls] + [flatten(c) for c in conds])


def examples_from_files(rootfiles, metapath, d, **kwargs):
    lines = load_lines(rootfiles, metapath, **kwargs)
    return expand_conditions(lines, d)


def fit_dicts(lines, d):
    lang_d, *conds_d = d
    ls, cs = zip(*lines)
    lang_d.fit(ls)
    for d, c in zip(conds_d, zip(*cs)):
        d.fit([c])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--cond_emb_dim', default=50)
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

    print("Processing datasets...")
    filenames = [os.path.join(args.path, f) for f in os.listdir(args.path)]
    lines = load_lines(filenames, args.metapath)
    nconds = len(next(lines)[1])
    lang_d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                  eos_token=u.EOS, bos_token=u.BOS)
    conds_d = []
    for _ in range(nconds):
        conds_d.append(Dict(sequential=False, force_unk=False))

    d = [lang_d] + conds_d

    print("Fitting dicts...")
    fit_dicts(lines, d)

    print(' * vocabulary size. %d' % len(lang_d))
    for idx, subd in enumerate(conds_d):
        print(' * condition [%d] with cardinality %d' % (idx, len(subd)))

    conds = [{'varnum': len(cond_d.vocab), 'emb_dim': args.cond_emb_dim}
             for cond_d in conds_d]

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

    files = len(os.listdir(args.path))
    for epoch in range(args.epochs):
        for batch_num, batch in enumerate(chunk(filenames, args.filebatch)):
            path = '/tmp/.seqmod_{filename}_{batch}.pt'.format(
                filename=os.path.basename(args.path), batch=batch_num)
            if epoch == 0:
                examples = examples_from_files(batch, args.metapath, d)
                u.save_model(examples, path)
            else:
                examples = u.load_model(path)
            train, valid, test = BlockDataset.splits_from_data(
                examples, d, args.batch_size, args.bptt, gpu=args.gpu, 
                test=args.test_split, dev=args.dev_split)

            trainer = CLMTrainer(
                model, {'train': train, 'valid': valid, 'test': test},
                criterion, optim)
            num_checkpoints = \
                len(train) // (args.checkpoint * args.hooks_per_epoch)
            trainer.add_loggers(std_logger, visdom_logger)
            trainer.train(1, args.checkpoint, gpu=args.gpu)
