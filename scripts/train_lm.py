
import os
import warnings

import random; random.seed(1001)

import torch
try:
    torch.cuda.manual_seed(1001)
except:
    warnings.warn('no NVIDIA driver found')
    torch.manual_seed(1001)

from seqmod.modules.lm import LM
from seqmod import utils as u

from seqmod.misc.trainer import Trainer
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.preprocess import text_processor
from seqmod.misc.early_stopping import EarlyStopping


# Load data
def load_lines(path, processor=text_processor()):
    lines = []
    if os.path.isfile(path):
        input_files = [path]
    else:
        input_files = [os.path.join(path, f) for f in os.listdir(path)]
    for path in input_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if processor is not None:
                    line = processor(line)
                if line:
                    lines.append(line)
    return lines


def load_from_file(path):
    if path.endswith('npy'):
        import numpy as np
        array = np.load(path).astype(np.int64)
        data = torch.LongTensor(array)
    elif path.endswith('.pt'):
        data = torch.load(path)
    else:
        raise ValueError('Unknown input format [%s]' % path)
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--att_dim', default=0, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    parser.add_argument('--maxouts', default=2, type=int)
    parser.add_argument('--train_init', action='store_true')
    # dataset
    parser.add_argument('--path', required=True)
    parser.add_argument('--processed', action='store_true')
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='token')
    parser.add_argument('--dev_split', default=0.1, type=float)
    parser.add_argument('--test_split', default=0.1, type=float)
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--gpu', action='store_true')
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=5, type=int)
    parser.add_argument('--decay_every', default=1, type=int)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    # - check
    parser.add_argument('--seed', default=None)
    parser.add_argument('--decoding_method', default='sample')
    parser.add_argument('--max_seq_len', default=25, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--hooks_per_epoch', default=5, type=int)
    parser.add_argument('--log_checkpoints', action='store_true')
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--visdom_host', default='localhost')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_path', default='models', type=str)
    args = parser.parse_args()

    if args.processed:
        print("Loading preprocessed datasets...")
        assert args.dict_path, "Processed data requires DICT_PATH"
        data, d = load_from_file(args.path), u.load_model(args.dict_path)
        train, test, valid = BlockDataset(
            data, d, args.batch_size, args.bptt, gpu=args.gpu, fitted=True
        ).splits(test=args.test_split, dev=args.dev_split)
        del data
    else:
        print("Processing datasets...")
        proc = text_processor(lower=args.lower, num=args.num, level=args.level)
        d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                 eos_token=u.EOS, force_unk=True)
        train, valid, test = None, None, None
        # already split
        if os.path.isfile(os.path.join(args.path, 'train.txt')):
            if not os.path.isfile(os.path.join(args.path, 'valid.txt')):
                raise ValueError("train.txt requires test.txt")
            train_data = load_lines(
                os.path.join(args.path, 'train.txt'), processor=proc)
            d.fit(train_data)
            train = BlockDataset(
                train_data, d, args.batch_size, args.bptt, gpu=args.gpu)
            del train_data
            test = BlockDataset(
                load_lines(os.path.join(args.path, 'test.txt'), proc),
                d, args.batch_size, args.bptt, gpu=args.gpu, evaluation=True)
            if os.path.isfile(os.path.join(args.path, 'valid.txt')):
                valid = BlockDataset(
                    load_lines(
                        os.path.join(args.path, 'valid.txt'), processor=proc),
                    d, args.batch_size, args.bptt, gpu=args.gpu,
                    evaluation=True)
            else:
                train, valid = train.splits(dev=None, test=args.dev_split)
        # do split, assume input is single file or dir with txt files
        else:
            data = load_lines(args.path, processor=proc)
            d.fit(data)
            train, valid, test = BlockDataset(
                data, d, args.batch_size, args.bptt, gpu=args.gpu
            ).splits(test=args.test_split, dev=args.dev_split)
            del data

    print(' * vocabulary size. %d' % len(d))
    print(' * number of train batches. %d' % len(train))

    print('Building model...')
    m = LM(len(d), args.emb_dim, args.hid_dim,
           num_layers=args.num_layers, cell=args.cell, dropout=args.dropout,
           att_dim=args.att_dim, tie_weights=args.tie_weights,
           deepout_layers=args.deepout_layers, train_init=args.train_init,
           deepout_act=args.deepout_act, maxouts=args.maxouts,
           word_dropout=args.word_dropout, target_code=d.get_unk())

    u.initialize_model(m)

    if args.gpu:
        m.cuda()

    print(m)
    print('* number of parameters: %d' % m.n_params())

    optim = Optimizer(
        m.parameters(), args.optim, lr=args.lr, max_norm=args.max_norm,
        lr_decay=args.lr_decay, start_decay_at=args.start_decay_at,
        decay_every=args.decay_every)

    # create trainer
    trainer = Trainer(m, {"train": train, "test": test, "valid": valid}, optim)

    # hooks
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(10, patience=args.early_stopping)
    model_hook = u.make_lm_hook(
        d, method=args.decoding_method, temperature=args.temperature,
        max_seq_len=args.max_seq_len, gpu=args.gpu,
        early_stopping=early_stopping)
    trainer.add_hook(model_hook, hooks_per_epoch=args.hooks_per_epoch)

    # loggers
    trainer.add_loggers(StdLogger())
    if args.visdom:
        visdom_logger = VisdomLogger(
            log_checkpoints=args.log_checkpoints, env='lm',
            server='http://' + args.visdom_host)
        trainer.add_loggers(visdom_logger)

    (best_model, valid_loss), test_loss = trainer.train(
        args.epochs, args.checkpoint)

    if args.save:
        args = vars(args)       # make dictionary args
        u.save_checkpoint(args.save_path, best_model, d, args, ppl=test_loss)
