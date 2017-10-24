
import os
import argparse

import torch
from torch.autograd import Variable

from seqmod import utils as u
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.preprocess import text_processor
from seqmod.misc.dataset import PairedDataset, Dict
from seqmod.misc.trainer import Trainer
from seqmod.loaders import load_twisty, load_dataset

from w2v import load_embeddings
from vae import SequenceVAE


def kl_weight_hook(trainer, epoch, batch, checkpoints):
    trainer.log("info", "kl weight: [%g]" % trainer.kl_weight)


def make_generate_hook(target="This is just a tweet and not much more", n=5):

    def hook(trainer, epoch, batch, checkpoints):
        d = trainer.datasets['train'].d['src']
        inp = torch.LongTensor([d.index(i) for i in target.split()])
        inp = Variable(inp, volatile=True).unsqueeze(1)
        z_params = trainer.model.encode(inp)
        for hyp_num in range(1, n + 1):
            score, hyp = trainer.model.generate(z_params=z_params)
            trainer.log("info", u.format_hyp(score[0], hyp[0], hyp_num, d))

    return hook


def load_lines(path, processor=text_processor()):
    lines = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if processor is not None:
                line = processor(line)
            if line:
                lines.append(line)
    return lines


def load_from_lines(path, batch_size, max_size=1000000, min_freq=5,
                    gpu=False, shuffle=True, **kwargs):
    lines = load_lines(path)

    ldict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
                 max_size=max_size, min_freq=min_freq).fit(lines)

    return PairedDataset(
        lines, None, {'src': ldict}, batch_size, gpu=gpu
    ).splits(shuffle=shuffle, **kwargs)


def load_penn(path, batch_size, max_size=1000000, min_freq=1,
              gpu=False, shuffle=True):
    train_data = load_lines(os.path.join(path, 'train.txt'))
    valid_data = load_lines(os.path.join(path, 'valid.txt'))
    test_data = load_lines(os.path.join(path, 'test.txt'))

    d = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
             max_size=max_size, min_freq=min_freq)
    d.fit(train_data, valid_data)

    train = PairedDataset(
        train_data, None, {'src': d}, batch_size, gpu=gpu)
    valid = PairedDataset(
        valid_data, None, {'src': d}, batch_size, gpu=gpu, evaluation=True)
    test = PairedDataset(
        test_data, None, {'src': d}, batch_size, gpu=gpu, evaluation=True)

    return train.sort_(), valid.sort_(), test.sort_()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--emb_dim', default=50, type=int)
    parser.add_argument('--hid_dim', default=50, type=int)
    parser.add_argument('--z_dim', default=50, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_init', action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--add_z', action='store_true')
    parser.add_argument('--load_embeddings', action='store_true')
    parser.add_argument('--flavor', default=None)
    parser.add_argument('--suffix', default=None)
    # training
    parser.add_argument('--optim', default='RMSprop')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_decay', default=0.85, type=float)
    parser.add_argument('--start_decay_at', default=1, type=int)
    parser.add_argument('--inflection_point', default=10000, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=564)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outputfile', default=None)
    parser.add_argument('--checkpoints', default=100, type=int)
    # dataset
    parser.add_argument('--source', required=True)
    parser.add_argument('--source_path')
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--max_size', default=50000, type=int)
    parser.add_argument('--level', default='token')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cache_data', action='store_true')
    args = parser.parse_args()

    prefix = '{source}.{level}.{min_len}.{min_freq}.{concat}.{max_size}' \
             .format(**vars(args))

    print("Loading data...")
    # preprocess
    if not args.cache_data or not os.path.isfile('data/%s_train.pt' % prefix):
        if args.source == 'twisty':
            src, trg = load_twisty(
                min_len=args.min_len, level=args.level, concat=args.concat,
                processor=text_processor(lower=False))
            train, test, valid = load_dataset(
                src, trg, args.batch_size,
                min_freq=args.min_freq, max_size=args.max_size,
                gpu=args.gpu, dev=args.dev, test=args.test)
        elif args.source == 'penn':
            train, test, valid = load_penn(
                "~/corpora/penn", args.batch_size,
                min_freq=args.min_freq, max_size=args.max_size, gpu=args.gpu)
        else:
            train, test, valid = load_from_lines(
                args.source_path, args.batch_size,
                min_freq=args.min_freq, max_size=args.max_size,
                gpu=args.gpu, dev=args.dev, test=args.text)
        # save
        if args.cache_data:
            train.to_disk('data/%s_train.pt' % prefix)
            test.to_disk('data/%s_test.pt' % prefix)
            valid.to_disk('data/%s_valid.pt' % prefix)
    # load from file
    else:
        train = PairedDataset.from_disk('data/%s_train.pt' % prefix)
        test = PairedDataset.from_disk('data/%s_test.pt' % prefix)
        valid = PairedDataset.from_disk('data/%s_valid.pt' % prefix)
        train.set_gpu(args.gpu)
        test.set_gpu(args.gpu)
        valid.set_gpu(args.gpu)
        train.set_batch_size(args.batch_size)
        test.set_batch_size(args.batch_size)
        valid.set_batch_size(args.batch_size)

    print("* Number of train batches %d" % len(train))

    print("Building model...")
    model = SequenceVAE(
        args.emb_dim, args.hid_dim, args.z_dim, train.d['src'],
        num_layers=args.num_layers, cell=args.cell, dropout=args.dropout,
        add_z=args.add_z, word_dropout=args.word_dropout,
        tie_weights=args.tie_weights, project_init=args.project_init,
        inflection_point=args.inflection_point)
    print(model)

    u.initialize_model(model)

    if args.load_embeddings:
        weight = load_embeddings(
            train.d['src'].vocab,
            args.flavor,
            args.suffix,
            '~/data/word_embeddings')
        model.init_embeddings(weight)

    if args.gpu:
        model.cuda()

    def on_lr_update(old_lr, new_lr):
        trainer.log("info", "Resetting lr [%g -> %g]" % (old_lr, new_lr))

    optimizer = Optimizer(
        model.parameters(), args.optim, lr=args.lr,
        max_norm=args.max_norm, weight_decay=args.weight_decay,
        # SGD-only
        start_decay_at=args.start_decay_at, lr_decay=args.lr_decay,
        on_lr_update=on_lr_update)

    class VAETrainer(Trainer):
        def on_batch_end(self, epoch, batch, loss):
            # reset kl weight
            total_batches = len(self.datasets['train'])
            self.model.kl_weight = self.model.kl_schedule(
                batch + total_batches * epoch)

    losses = [{'loss': 'log-loss'},
              {'loss': 'kl', 'format': lambda loss: loss}]

    trainer = VAETrainer(
        model, {'train': train, 'valid': valid, 'test': test}, optimizer,
        losses=losses)
    trainer.add_loggers(
        StdLogger(), VisdomLogger(env='vae', losses=('rec', 'kl'), max_y=600))

    trainer.train(args.epochs, args.checkpoints, shuffle=True)
