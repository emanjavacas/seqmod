
import os
import sys
import time

seed = 1001
import random                   # nopep8
random.seed(seed)

import torch                    # nopep8
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import torch.nn as nn           # nopep8

from lm import LM               # nopep8
from trainer import LMTrainer, Logger   # nopep8
from optimizer import Optimizer         # nopep8
from dataset import Dict, BlockDataset  # nopep8
from preprocess import text_processor   # nopep8
from early_stopping import EarlyStopping  # nopep8
import utils as u                         # nopep8


# Load data
def load_lines(path, processor=text_processor()):
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if processor is not None:
                line = processor(line)
            line = line.split()
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


# hook
def make_lm_check_hook(d, gpu, early_stopping):

    def hook(trainer, batch_num, checkpoint):
        print("Checking training...")
        loss = trainer.validate_model()
        print("Valid loss: %g" % loss)
        print("Registering early stopping loss...")
        if early_stopping is not None:
            early_stopping.add_checkpoint(loss)
        print("Generating text...")
        print("***")
        scores, hyps = trainer.model.generate_beam(
            d.get_bos(), d.get_eos(), gpu=gpu, max_seq_len=100)
        u.print_hypotheses(scores, hyps, d)
        print("***")

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--processed', action='store_true',
                        help='Is data in processed format?')
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_on_tied_weights', action='store_true')
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--checkpoints_per_epoch', default=5, type=int)
    parser.add_argument('--optim', default='RMSprop', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=5, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--prefix', default='model', type=str)
    args = parser.parse_args()

    if args.processed:
        print("Loading preprocessed datasets...")
        assert args.dict_path, "Processed data requires DICT_PATH"
        data, d = load_from_file(args.path), u.load_model(args.dict_path)
        train, test, valid = BlockDataset(
            data, d, args.batch_size, args.bptt, gpu=args.gpu, fitted=True
        ).splits(test=0.1, dev=0.1)
        del data
    else:
        print("Processing datasets...")
        train_data = load_lines(args.path + 'train.txt')
        valid_data = load_lines(args.path + 'valid.txt')
        test_data = load_lines(args.path + 'test.txt')
        d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                 eos_token=u.EOS, bos_token=u.EOS)
        d.fit(train_data, valid_data, test_data)
        train = BlockDataset(
            train_data, d, args.batch_size, args.bptt, gpu=args.gpu)
        valid = BlockDataset(
            valid_data, d, args.batch_size, args.bptt, gpu=args.gpu,
            evaluation=True)
        test = BlockDataset(
            test_data, d, args.batch_size, args.bptt, gpu=args.gpu,
            evaluation=True)
        del train_data, valid_data, test_data

    print(' * vocabulary size. %d' % len(d))
    print(' * number of train batches. %d' % len(train))

    print('Building model...')
    model = LM(len(d), args.emb_dim, args.hid_dim,
               num_layers=args.layers, cell=args.cell,
               dropout=args.dropout, tie_weights=args.tie_weights,
               project_on_tied_weights=args.project_on_tied_weights)

    model.apply(u.make_initializer())
    if args.gpu:
        model.cuda()

    print(model)
    print('* number of parameters: %d' % model.n_params())

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = nn.CrossEntropyLoss()

    datasets = {"train": train, "test": test, "valid": valid}
    trainer = LMTrainer(model, datasets, criterion, optim)

    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(args.early_stopping)
    model_check_hook = make_lm_check_hook(
        d, args.gpu, early_stopping=early_stopping)
    num_checks = len(train) // (args.checkpoint * args.checkpoints_per_epoch)
    trainer.add_hook(model_check_hook, num_checkpoints=num_checks)

    trainer.add_loggers(Logger())

    trainer.train(args.epochs, args.checkpoint, gpu=args.gpu)

    if args.save:
        test_ppl = trainer.validate_model(test=True)
        print("Test perplexity: %g" % test_ppl)
        if args.save:
            f = '{prefix}.{cell}.{layers}l.{hid_dim}h.{emb_dim}e.{bptt}b.{ppl}'
            fname = f.format(ppl="%.2f" % test_ppl, **vars(args))
            if os.path.isfile(fname):
                answer = input("File [%s] exists. Overwrite? (y/n): " % fname)
                if answer.lower() not in ("y", "yes"):
                    print("Goodbye!")
                    sys.exit(0)
            print("Saving model to [%s]..." % fname)
            u.save_model(model, fname, d=d)
