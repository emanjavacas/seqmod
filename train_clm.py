
import os

import random; random.seed(1001)

import torch
try:
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')
    torch.manual_seed(1001)

import torch.nn as nn

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
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=48, type=int)
    parser.add_argument('--cond_emb_dim', default=24)
    parser.add_argument('--hid_dim', default=640, type=int)
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
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='token')
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--bptt', default=150, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--dev_split', type=float, default=0.05)
    parser.add_argument('--load_dict', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--save_dict', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--dict_path')
    parser.add_argument('--model_path')
    parser.add_argument('--save_data_prefix',
                        help='Prefix for caching preprocessed batches')
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
    if args.load_dict:
        print('Loading dicts...')
        assert args.dict_path, "load_dict requires dict_path"
        d = u.load_model(args.dict_path)
        lang_d, *conds_d = d
    else:
        print("Fitting dicts...")
        # lang Dict
        lang_d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                      eos_token=u.EOS, bos_token=u.BOS)
        # cond Dicts
        lines = load_lines(filenames, args.metapath)
        nconds = len(next(lines)[1])
        conds_d = [Dict(sequential=False, force_unk=False)
                   for _ in range(nconds)]
        d = [lang_d] + conds_d
        ls, cs = zip(*lines)
        print("Fitting language Dict")
        lang_d.fit(ls)
        print("Fitting condition Dicts")
        for d, c in zip(conds_d, zip(*cs)):
            d.fit([c])
        if args.save_dict:
            print("Saving dicts")
            assert args.dict_path, "save_dict requires dict_path"
            u.save_model(d, args.dict_path)

    # conditional structure
    conds = []
    print(' * vocabulary size. %d' % len(lang_d))
    # conditional structure
    conds = []
    for idx, subd in enumerate(conds_d):
        print(' * condition [%d] with cardinality %d' % (idx, len(subd)))
        conds.append({'varnum': len(subd), 'emb_dim': args.cond_emb_dim})

    if args.load_model:
        print('Loading model...')
        assert args.model_path, "load_model requires model_path"
        model = u.load_model(args.model_path)
    else:
        print('Building model...')
        model = LM(len(lang_d), args.emb_dim, args.hid_dim,
                   num_layers=args.layers, cell=args.cell,
                   dropout=args.dropout, tie_weights=args.tie_weights,
                   deepout_layers=args.deepout_layers,
                   deepout_act=args.deepout_act,
                   word_dropout=args.word_dropout,
                   target_code=lang_d.get_unk(), conds=conds)
        model.apply(u.make_initializer())
    print(model)
    print(' * n parameters. %d' % model.n_params())

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
    check_hook = u.make_clm_hook(
        d, max_seq_len=args.max_seq_len, samples=5, gpu=args.gpu,
        method=args.decoding_method, temperature=args.temperature)

    # loggers
    visdom_logger = VisdomLogger(
        log_checkpoints=args.log_checkpoints, title=args.prefix, env='clm',
        server='http://' + args.visdom_server)
    std_logger = StdLogger()

    trainer = CLMTrainer(
        model, {'train': train, 'valid': valid, 'test': test},
        criterion, optim)
    trainer.add_loggers(std_logger, visdom_logger)
    trainer.add_hook(check_hook, hooks_per_epoch=args.hooks_per_epoch)
    trainer.train(args.epochs, args.checkpoint, gpu=args.gpu)
