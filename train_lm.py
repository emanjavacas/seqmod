
import os
import sys

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

from seqmod.modules.lm import LM       # nopep8
from seqmod import utils as u  # nopep8

from seqmod.misc.trainer import LMTrainer               # nopep8
from seqmod.misc.loggers import StdLogger, VisdomLogger  # nopep8
from seqmod.misc.optimizer import Optimizer              # nopep8
from seqmod.misc.dataset import Dict, BlockDataset       # nopep8
from seqmod.misc.preprocess import text_processor        # nopep8
from seqmod.misc.early_stopping import EarlyStopping     # nopep8


# Load data
def load_lines(path, processor=text_processor()):
    lines = []
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


# hook
def make_lm_check_hook(d, seed_text, max_seq_len=25, gpu=False,
                       method='sample', temperature=1, width=5,
                       early_stopping=None, validate=True):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Checking training...")
        if validate:
            loss = trainer.validate_model()
            trainer.log("info", "Valid loss: %g" % loss)
            trainer.log("info", "Registering early stopping loss...")
            if early_stopping is not None:
                early_stopping.add_checkpoint(loss)
        trainer.log("info", "Generating text...")
        scores, hyps = trainer.model.generate(
            d, seed_text=seed_text, max_seq_len=max_seq_len, gpu=gpu,
            method=method, temperature=temperature, width=width)
        hyps = [u.format_hyp(score, hyp, hyp_num + 1, d)
                for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
        trainer.log("info", '\n***' + ''.join(hyps) + "\n***")

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--att_dim', default=0, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    # dataset
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
    parser.add_argument('--optim', default='RMSprop', type=str)
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
        assert args.dict_path, "Processed data requires DICT_PATH"
        data, d = load_from_file(args.path), u.load_model(args.dict_path)
        train, test, valid = BlockDataset(
            data, d, args.batch_size, args.bptt, gpu=args.gpu, fitted=True
        ).splits(test=0.1, dev=0.1)
        del data
    else:
        print("Processing datasets...")
        proc = text_processor(
            lower=args.lower, num=args.num, level=args.level)
        train_data = load_lines(args.path + 'train.txt', processor=proc)
        valid_data = load_lines(args.path + 'valid.txt', processor=proc)
        test_data = load_lines(args.path + 'test.txt', processor=proc)
        d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                 eos_token=u.EOS, bos_token=u.BOS)
        d.fit(train_data, valid_data)
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
               num_layers=args.layers, cell=args.cell, dropout=args.dropout,
               att_dim=args.att_dim, tie_weights=args.tie_weights,
               deepout_layers=args.deepout_layers,
               deepout_act=args.deepout_act, word_dropout=args.word_dropout,
               target_code=d.get_unk())

    model.apply(u.make_initializer())
    if args.gpu:
        model.cuda()

    print(model)
    print('* number of parameters: %d' % model.n_params())

    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at,
        decay_every=args.decay_every)
    criterion = nn.CrossEntropyLoss()

    # create trainer
    trainer = LMTrainer(model, {"train": train, "test": test, "valid": valid},
                        criterion, optim)

    # hooks
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(args.early_stopping)
    model_check_hook = make_lm_check_hook(
        d, method=args.decoding_method, temperature=args.temperature,
        max_seq_len=args.max_seq_len, seed_text=args.seed, gpu=args.gpu,
        early_stopping=early_stopping)
    num_checkpoints = len(train) // (args.checkpoint * args.hooks_per_epoch)
    trainer.add_hook(model_check_hook, num_checkpoints=num_checkpoints)

    # loggers
    visdom_logger = VisdomLogger(
        log_checkpoints=args.log_checkpoints, title=args.prefix, env='lm',
        server='http://' + args.visdom_server)
    trainer.add_loggers(StdLogger(), visdom_logger)

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
