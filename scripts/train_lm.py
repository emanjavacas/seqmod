
import os
import warnings

import random; random.seed(1001)

import torch
try:
    torch.cuda.manual_seed(1001)
except:
    warnings.warn('no NVIDIA driver found')
    torch.manual_seed(1001)

from torch import optim

from seqmod.modules.lm import LM
from seqmod import utils as u

from seqmod.misc import Trainer, StdLogger, VisdomLogger
from seqmod.misc import Dict, BlockDataset, text_processor
from seqmod.misc import EarlyStopping


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


def load_dataset(path, d, processor, args):
    data = load_lines(path, processor=processor)
    if not d.fitted:
        d.fit(data)

    return BlockDataset(data, d, args.batch_size, args.bptt, gpu=args.gpu)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--att_dim', default=0, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
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
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=35, type=int)
    parser.add_argument('--gpu', action='store_true')
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    # - check
    parser.add_argument('--seed', default=None)
    parser.add_argument('--decoding_method', default='sample')
    parser.add_argument('--max_seq_len', default=25, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--checkpoint', default=100, type=int)
    parser.add_argument('--hooks_per_epoch', default=1, type=int)
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

    else:
        print("Processing datasets...")
        processor = text_processor(
            lower=args.lower, num=args.num, level=args.level)
        d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                 eos_token=u.EOS, force_unk=True)

        # already split
        if os.path.isfile(os.path.join(args.path, 'train.txt')):
            # train set
            path = os.path.join(args.path, 'train.txt')
            train = load_dataset(path, d, processor, args)
            # test set
            path = os.path.join(args.path, 'test.txt')
            test = load_dataset(path, d, processor, args)
            # valid set
            if os.path.isfile(os.path.join(args.path, 'valid.txt')):
                path = os.path.join(args.path, 'valid.txt')
                valid = load_dataset(path, d, processor, args)
            else:
                train, valid = train.splits(dev=None, test=args.dev_split)

        # split, assume input is single file or dir with txt files
        else:
            data = load_lines(args.path, processor=processor)
            d.fit(data)
            train, valid, test = BlockDataset(
                data, d, args.batch_size, args.bptt, gpu=args.gpu
            ).splits(test=args.test_split, dev=args.dev_split)

    print(' * vocabulary size. {}'.format(len(d)))
    print(' * number of train batches. {}'.format(len(train)))

    print('Building model...')
    m = LM(len(d), args.emb_dim, args.hid_dim, d,
           num_layers=args.num_layers, cell=args.cell, dropout=args.dropout,
           att_dim=args.att_dim, tie_weights=args.tie_weights,
           deepout_layers=args.deepout_layers, train_init=args.train_init,
           deepout_act=args.deepout_act, maxouts=args.maxouts,
           word_dropout=args.word_dropout)

    u.initialize_model(m, rnn={'type': 'orthogonal', 'args': {'gain': 1.0}})

    if args.gpu:
        m.cuda()

    print(m)
    print('* number of parameters: {}'.format(m.n_params()))

    optimizer = getattr(optim, args.optim)(m.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.5)

    # create trainer
    trainer = Trainer(
        m, {"train": train, "test": test, "valid": valid}, optimizer,
        max_norm=args.max_norm, scheduler=scheduler)

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
        u.save_checkpoint(
            args.save_path, best_model, d, vars(args), ppl=test_loss)
