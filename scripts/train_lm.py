
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
from seqmod.misc import Trainer, StdLogger, VisdomLogger, EarlyStopping
from seqmod.misc import Dict, BlockDataset, text_processor, Checkpoint
from seqmod.misc import inflection_sigmoid, inverse_exponential, inverse_linear
from seqmod import utils as u
from seqmod.loaders import load_lines


# Load data
def load_from_file(path):
    if path.endswith('npy') or path.endswith('npz'):
        import numpy as np
        array = np.load(path).astype(np.int64)
        data = torch.tensor(array)
    elif path.endswith('.pt'):
        data = torch.load(path)
    else:
        raise ValueError('Unknown input format [%s]' % path)
    return data


def load_dataset(path, d, processor, args):
    data = list(load_lines(path, processor=processor))
    if not d.fitted:
        d.fit(data)

    return BlockDataset(data, d, args.batch_size, args.bptt, device=args.device)


def make_lr_hook(optimizer, factor, patience, threshold=0.05, verbose=True):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, threshold=threshold,
        verbose=verbose)

    def hook(trainer, epoch, batch_num, checkpoint):
        loss = trainer.validate_model()
        if verbose:
            trainer.log("validation_end", {"epoch": epoch, "loss": loss.pack()})
        scheduler.step(loss.reduce())

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=1000, type=int)
    parser.add_argument('--att_dim', default=0, type=int)
    parser.add_argument('--dropout', default=0.6, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--mixtures', default=0, type=int)
    parser.add_argument('--sampled_softmax', action='store_true')
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
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--use_schedule', action='store_true')
    parser.add_argument('--schedule_init', type=float, default=1.0)
    parser.add_argument('--schedule_inflection', type=int, default=2)
    parser.add_argument('--schedule_steepness', type=float, default=1.75)
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--patience', default=0, type=int)
    parser.add_argument('--lr_schedule_checkpoints', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--lr_checkpoints_per_epoch', type=int, default=1)
    # - check
    parser.add_argument('--seed', default=None)
    parser.add_argument('--max_seq_len', default=25, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--checkpoint', default=100, type=int)
    parser.add_argument('--hooks_per_epoch', default=1, type=int)
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--visdom_host', default='localhost')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.processed:
        print("Loading preprocessed datasets...")
        assert args.dict_path, "Processed data requires DICT_PATH"
        d = u.load_model(args.dict_path)

        if os.path.isfile(args.path):
            # single files full dataset
            train, test, valid = BlockDataset(
                load_from_file(args.path), d, args.batch_size, args.bptt,
                device=args.device, fitted=True
            ).splits(test=args.test_split, dev=args.dev_split)
            train = BlockDataset(load_from_file(args.path), )
        else:
            # assume path is prefix to train/test splits
            train = load_from_file(args.path + '.train.npz')
            if os.path.isfile(args.path + '.test.npz'):
                train, valid = BlockDataset(
                    train, d, args.batch_size, args.bptt,
                    device=args.device, fitted=True
                ).splits(test=args.dev_split, dev=None)
                test = BlockDataset(
                    load_from_file(args.path + '.test.npz'), d,
                    args.batch_size, args.bptt, device=args.device, fitted=True)
            else:
                train, valid, test = BlockDataset(
                    train, d, args.batch_size, args.bptt,
                    device=args.device, fitted=True
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
            data = list(load_lines(args.path, processor=processor))
            d.fit(data)
            train, valid, test = BlockDataset(
                data, d, args.batch_size, args.bptt, device=args.device,
            ).splits(test=args.test_split, dev=args.dev_split)

    print(' * vocabulary size. {}'.format(len(d)))
    print(' * number of train batches. {}'.format(len(train)))

    print('Building model...')
    m = LM(args.emb_dim, args.hid_dim, d, exposure_rate=args.schedule_init,
           num_layers=args.num_layers, cell=args.cell, dropout=args.dropout,
           att_dim=args.att_dim, tie_weights=args.tie_weights, mixtures=args.mixtures,
           deepout_layers=args.deepout_layers, train_init=args.train_init,
           deepout_act=args.deepout_act, maxouts=args.maxouts,
           sampled_softmax=args.sampled_softmax, word_dropout=args.word_dropout)

    u.initialize_model(
        m,
        rnn={'type': 'orthogonal_', 'args': {'gain': 1.0}},
        emb={'type': 'uniform_', 'args': {'a': -0.1, 'b': 0.1}})

    m.to(device=args.device)

    print(m)
    print('* number of parameters: {}'.format(m.n_params()))

    if args.optim == 'Adam':
        optimizer = getattr(optim, args.optim)(
            m.parameters(), lr=args.lr, betas=(0., 0.99), eps=1e-5)
    else:
        optimizer = getattr(optim, args.optim)(m.parameters(), lr=args.lr)

    # create trainer
    loss_type = 'bpc' if args.level == 'char' else 'ppl'
    trainer = Trainer(
        m, {"train": train, "test": test, "valid": valid}, optimizer,
        max_norm=args.max_norm, losses=(loss_type,))

    # hooks
    # - general hook
    early_stopping = None
    if args.patience > 0:
        early_stopping = EarlyStopping(args.patience)

    checkpoint = None
    if args.save:
        checkpoint = Checkpoint(m.__class__.__name__, keep=3).setup(args)

    model_hook = u.make_lm_hook(
        d, temperature=args.temperature, max_seq_len=args.max_seq_len,
        device=args.device, level=args.level, early_stopping=early_stopping,
        checkpoint=checkpoint)
    trainer.add_hook(model_hook, hooks_per_epoch=args.hooks_per_epoch)

    # - scheduled sampling hook
    if args.use_schedule:
        schedule = inflection_sigmoid(
            len(train) * args.schedule_inflection, args.schedule_steepness,
            a=args.schedule_init, inverse=True)
        trainer.add_hook(
            u.make_schedule_hook(schedule, verbose=True), hooks_per_epoch=10e4)

    # - lr schedule hook
    if args.lr_schedule_factor < 1.0:
        hook = make_lr_hook(
            optimizer, args.lr_schedule_factor, args.lr_schedule_checkpoints)
        # run a hook args.checkpoint * 4 batches
        trainer.add_hook(hook, hooks_per_epoch=args.lr_checkpoints_per_epoch)

    # loggers
    trainer.add_loggers(StdLogger())
    if args.visdom:
        visdom_logger = VisdomLogger(
            env='lm', server='http://' + args.visdom_host)
        trainer.add_loggers(visdom_logger)

    (best_model, valid_loss), test_loss = trainer.train(
        args.epochs, args.checkpoint, use_schedule=args.use_schedule)
