
import os

import torch
import torch.nn as nn

from seqmod.modules.lm import LM
from seqmod.misc.dataset import BlockDataset, Dict, CompressionTable
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.early_stopping import EarlyStopping
from seqmod.misc.trainer import CLMTrainer
from seqmod.misc.loggers import StdLogger
import seqmod.utils as u


def readlines(inputfile):
    with open(inputfile, 'r', newline='\n') as f:
        for line in f:
            *labels, sent = line.split('\t')
            yield labels, sent


def linearize_data(lines, conds, lang_d, conds_d, table=None):
    for line, line_conds in zip(lines, conds):
        line_conds = tuple(d.index(c) for d, c in zip(conds_d, line_conds))
        for char in next(lang_d.transform([line])):
            yield char
            if table is None:
                for c in line_conds:
                    yield c
            else:
                yield table.hash_vals(line_conds)


def examples_from_lines(lines, conds, lang_d, conds_d, table=None):
    generator = linearize_data(lines, conds, lang_d, conds_d, table=table)
    dims = 2 if table is not None else len(conds_d) + 1
    return torch.LongTensor(list(generator)).view(-1, dims).t().contiguous()


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
    parser.add_argument('--path')
    parser.add_argument('--processed', action='store_true')
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='token')
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--bptt', default=150, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--dev_split', type=float, default=0.05)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_path', default='./')
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--data_path')
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
    args = parser.parse_args()

    if args.load_data:
        train, test, d, table = u.load_model(args.data_path)
        lang_d, *conds_d = d
    else:
        print("Fitting dictionaries")
        lang_d = Dict(
            max_size=args.max_size, min_freq=args.min_freq, eos_token=u.EOS)
        conds_d = [Dict(sequential=False, force_unk=False) for _ in range(2)]
        linesiter = readlines(os.path.join(args.path, 'train.csv'))
        train_labels, train_lines = zip(*linesiter)
        print("Fitting language Dict")
        lang_d.fit(train_lines)
        print("Fitting condition Dicts")
        for d, cond in zip(conds_d, zip(*train_labels)):
            d.fit([cond])

        print("Processing datasets")
        print("Processing train")
        table = CompressionTable(len(conds_d))
        train = examples_from_lines(
            train_lines, train_labels, lang_d, conds_d, table=table)
        del train_lines, train_conds
        print("Processing test")
        linesiter = readlines(os.path.join(args.path, 'test.csv'))
        test_labels, test_lines = zip(*linesiter)
        test = examples_from_lines(
            test_lines, test_labels, lang_d, conds_d, table=table)
        del test_lines, test_labels
        d = tuple([lang_d] + conds_d)

        if args.save_data:
            assert args.data_path, "save_data requires data_path"
            u.save_model((train, test, d, table), args.data_path)

    train, valid = BlockDataset.splits_from_data(
        tuple(train), d, args.batch_size,
        args.bptt, gpu=args.gpu, table=table,
        test=None, dev=args.dev_split)

    test = BlockDataset(
        tuple(test), d, args.batch_size, args.bptt,
        fitted=True, gpu=args.gpu, table=table)

    # conditional structure
    conds = []
    print(' * vocabulary size. %d' % len(lang_d))
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
        u.initialize_model(model)

    print(model)
    print(' * n parameters. %d' % model.n_params())

    if args.gpu:
        model.cuda()

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at,
        decay_every=args.decay_every)
    early_stopping = EarlyStopping(max(args.patience, 10), args.patience)
    criterion = nn.NLLLoss()

    # hook
    check_hook = u.make_clm_hook(
        d, max_seq_len=args.max_seq_len, gpu=args.gpu, sampled_conds=5,
        method=args.decoding_method, temperature=args.temperature)
    # logger
    std_logger = StdLogger()
    # trainer
    trainer = CLMTrainer(
        model, {'train': train, 'valid': valid, 'test': test},
        criterion, optim, early_stopping=early_stopping)
    trainer.add_loggers(std_logger)
    trainer.add_hook(check_hook, hooks_per_epoch=args.hooks_per_epoch)

    (best_model, val_ppl), test_ppl = trainer.train(
        args.epochs, args.checkpoint, gpu=args.gpu)

    if args.save:
        u.save_checkpoint(args.model_path, best_model, d, vars(args),
                          ppl=test_ppl)
