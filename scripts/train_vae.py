
import os
import random
import argparse

import torch
from torch.autograd import Variable

from seqmod import utils as u
from seqmod.misc import StdLogger, VisdomLogger, Optimizer
from seqmod.misc import text_processor, PairedDataset, Dict
from seqmod.misc import Trainer, EarlyStopping
from seqmod.modules.vae import SequenceVAE, kl_sigmoid_annealing_schedule
from seqmod.loaders import load_twisty, load_dataset

from w2v import load_embeddings


def kl_weight_hook(trainer, epoch, batch, checkpoints):
    trainer.log("info", "kl weight: [{:.3f}]".format(trainer.model.kl_weight))


def decode_report(trainer, src, z_params, n=5, keep=2):
    """
    src: LongTensor (seq_len, n)
    z_params (mu, logvar): FloatTensors (n * keep x z_dim)
    """
    d = trainer.datasets['train'].d['src']
    src = src.chunk(src.size(1), 1)
    z_params = zip(z_params[0].chunk(n, 0), z_params[1].chunk(n, 0))

    for idx, (src, z_params) in enumerate(zip(src, z_params)):
        # src: (1 x seq_len); z_params = (mu, logvar): ((keep x z_dim), ...)
        scores, hyps = trainer.model.generate(z_params=z_params)
        
        # build report
        report = '{}\nSource: '.format(idx)
        report += ' '.join(d.vocab[c] for c in src.squeeze().data.tolist())

        # report best n hypotheses
        report += '\nHypotheses:'
        report += "".join(
            u.format_hyp(scores[i], hyps[i], i, d) for i in range(len(hyps)))
        report += '\n\n'

        yield report


def make_generate_hook(n=5, keep=2):

    def hook(trainer, epoch, batch, checkpoints):
        # grab random batch from valid
        src, _ = valid[random.randint(0, len(valid)-1)]
        # grab random examples from batch
        idxs = torch.randperm(n)
        src = src[:, idxs.cuda() if src.data.is_cuda else idxs]
        emb = trainer.model.embeddings(src)
        mu, logvar = trainer.model.encoder(emb)
        z_params = mu.repeat(keep, 1), logvar.repeat(keep, 1)
        for report in decode_report(trainer, src, z_params, n=n, keep=keep):
            trainer.log("info", report)

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


def load_split_data(path, batch_size, max_size, min_freq, gpu):
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
    parser.add_argument('--emb_dim', default=100, type=int)
    parser.add_argument('--hid_dim', default=150, type=int)
    parser.add_argument('--z_dim', default=150, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_init', action='store_true')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--dont_add_z', action='store_true')
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
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--inflection', default=6000, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outputfile', default=None)
    parser.add_argument('--checkpoints', default=50, type=int)
    # dataset
    parser.add_argument('--source', default='wikitext-2',
                        help='Directory with split data')
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--max_tweets', default=0, type=int)
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--max_size', default=50000, type=int)
    parser.add_argument('--level', default='token')
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--cache_data', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    prefix = '{source}.{level}.{min_len}.{min_freq}.{concat}.{max_size}' \
             .format(**vars(args))

    # preprocess
    if not args.cache_data or not os.path.isfile('data/{}_train.pt'.format(prefix)):
        if args.source == 'twisty':
            src, trg = load_twisty(
                min_len=args.min_len, concat=args.concat,
                processor=text_processor(lower=False, level=args.level))
            train, test, valid = load_dataset(
                src, trg, args.batch_size,
                min_freq=args.min_freq, max_size=args.max_size,
                gpu=args.gpu, dev=args.dev, test=args.test,
            max_tweets=None if args.max_tweets == 0 else args.max_tweets)
        else:
            train, test, valid = load_split_data(
                os.path.join('~/corpora', args.source), args.batch_size,
                args.max_size, args.min_freq, args.gpu)
        # save
        if args.cache_data:
            train.to_disk('data/{}_train.pt'.format(prefix))
            test.to_disk('data/{}_test.pt'.format(prefix))
            valid.to_disk('data/{}_valid.pt'.format(prefix))

    # load from file
    else:
        train = PairedDataset.from_disk('data/{}_train.pt'.format(prefix))
        test = PairedDataset.from_disk('data/{}_test.pt'.format(prefix))
        valid = PairedDataset.from_disk('data/{}_valid.pt'.format(prefix))
        train.set_gpu(args.gpu)
        test.set_gpu(args.gpu)
        valid.set_gpu(args.gpu)
        train.set_batch_size(args.batch_size)
        test.set_batch_size(args.batch_size)
        valid.set_batch_size(args.batch_size)

    print("* Number of train batches {}".format(len(train)))

    print("Building model...")
    model = SequenceVAE(
        args.emb_dim, args.hid_dim, args.z_dim, train.d['src'],
        num_layers=args.num_layers, cell=args.cell, dropout=args.dropout,
        add_z=not args.dont_add_z, word_dropout=args.word_dropout,
        tie_weights=args.tie_weights, project_init=args.project_init,
        kl_schedule=kl_sigmoid_annealing_schedule(inflection=args.inflection))
    print(model)

    print("* number of parameters: {}".format(
        sum(p.nelement() for p in model.parameters())))

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
        trainer.log("info", "Resetting lr [{} -> {}]".format(old_lr, new_lr))

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

    losses = [{'loss': 'rec'},
              {'loss': 'kl', 'format': lambda loss: loss}]

    trainer = VAETrainer(
        model, {'train': train, 'valid': valid, 'test': test}, optimizer,
        losses=losses, early_stopping=EarlyStopping(5, patience=args.patience))
    trainer.add_hook(make_generate_hook(), hooks_per_epoch=1)
    trainer.add_hook(kl_weight_hook, hooks_per_epoch=2)
    trainer.add_loggers(
        StdLogger(), VisdomLogger(env='vae', losses=('rec', 'kl'), max_y=600))

    trainer.train(args.epochs, args.checkpoints, shuffle=True)
