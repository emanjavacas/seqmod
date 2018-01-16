
import os
import random
import argparse

import torch
from torch import optim

from seqmod import utils as u
from seqmod.misc import StdLogger, VisdomLogger
from seqmod.misc import text_processor, PairedDataset, Dict
from seqmod.misc import Trainer, EarlyStopping, inflection_sigmoid
from seqmod.modules.vae import kl_sigmoid_annealing_schedule
from seqmod.modules.vae import make_vae_encoder_decoder
from seqmod.loaders import load_twisty, load_split_data


def kl_weight_hook(trainer, epoch, batch, checkpoints):
    weight = trainer.model.encoder.kl_weight
    trainer.log("info", "KL weight: [{:.3f}]".format(weight))


def make_generate_hook(level, n=2, samples=2, beam_width=5):

    def hook(trainer, epoch, batch, checkpoints):
        # grab random batch from valid
        src, _ = valid[random.randint(0, len(valid)-1)]
        if trainer.model.encoder.conditional:
            (src, *_) = src
        (src, lengths) = src
        # grab random examples from batch
        idxs = torch.randperm(n)
        src = src[:, idxs.cuda() if src.data.is_cuda else idxs]
        # dict
        d = trainer.model.decoder.embeddings.d
        sep = ' ' if level == 'word' or level == 'token' else ''

        for idx, src in enumerate(src.chunk(src.size(1), 1)):
            report = '{}\nSource: '.format(idx + 1)
            report += sep.join(d.vocab[char.data[0]] for char in src)
            report += '\n'

            for sample in range(samples):
                scores, hyps, _ = trainer.model.translate_beam(
                    src, lengths=lengths, beam_width=beam_width)
                # select only best
                scores, hyps = [scores[0]], [hyps[0]]

                # report
                report += 'Sample {}:'.format(sample + 1)
                # report best n hypotheses
                report += "".join(
                    u.format_hyp(scores[i], hyps[i], i, d, level)
                    for i in range(len(hyps))
                ) + '\n'

            trainer.log("info", report)

    return hook


def load_twisty_dataset(src, trg, batch_size, max_size=100000, min_freq=5,
                        gpu=False, shuffle=True, **kwargs):
    """
    Wrapper function for twisty with sensible, overwritable defaults
    """
    tweets_dict = Dict(pad_token=u.PAD, eos_token=u.EOS,
                       bos_token=u.BOS, max_size=max_size, min_freq=min_freq)
    labels_dict = Dict(sequential=False, force_unk=False)
    tweets_dict.fit(src)
    labels_dict.fit(trg)
    d = {'src': tweets_dict, 'trg': labels_dict}
    splits = PairedDataset(src, trg, d, batch_size, gpu=gpu).splits(
        shuffle=shuffle, **kwargs)
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--emb_dim', default=100, type=int)
    parser.add_argument('--hid_dim', default=150, type=int)
    parser.add_argument('--z_dim', default=150, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--add_init_jitter', action='store_true')
    parser.add_argument('--encoder-summary', default='inner-attention')
    parser.add_argument('--deepout_layers', type=int, default=0)
    parser.add_argument('--deepout_act', default='ReLU')
    parser.add_argument('--dont_add_z', action='store_true')
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path')
    # training
    parser.add_argument('--optim', default='RMSprop')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--inflection', default=6000, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--outputfile', default=None)
    parser.add_argument('--use_schedule', action='store_true')
    parser.add_argument('--checkpoints', default=50, type=int)
    parser.add_argument('--hooks_per_epoch', default=2, type=int)
    # dataset
    parser.add_argument('--source', default='enwiki8',
                        help='Directory with split data')
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--test', default=0.2, type=float)
    parser.add_argument('--max_tweets', default=0, type=int)
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--max_len', default=200, type=int)
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

        processor = text_processor(lower=False, level=args.level)

        if args.source == 'twisty':
            src, trg = load_twisty(
                min_len=args.min_len, concat=args.concat, processor=processor)
            train, test, valid = load_twisty_dataset(
                src, trg, args.batch_size,
                min_freq=args.min_freq, max_size=args.max_size,
                gpu=args.gpu, dev=args.dev, test=args.test,
                max_tweets=None if args.max_tweets == 0 else args.max_tweets)
        else:
            train, test, valid = load_split_data(
                os.path.join('~/corpora', args.source), args.batch_size,
                args.max_size, args.min_freq, args.max_len, args.gpu, processor)
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
    model = make_vae_encoder_decoder(
        args.z_dim, args.num_layers, args.emb_dim, args.hid_dim,
        train.d['src'], cell=args.cell, encoder_summary=args.encoder_summary,
        dropout=args.dropout, word_dropout=args.word_dropout,
        add_z=not args.dont_add_z, tie_weights=args.tie_weights,
        deepout_layers=args.deepout_layers, deepout_act=args.deepout_act,
        train_init=args.train_init, add_init_jitter=args.add_init_jitter)
    print(model)

    print("* number of parameters: {}".format(
        sum(p.nelement() for p in model.parameters())))

    u.initialize_model(model)

    if args.init_embeddings:
        model.encoder.embeddings.init_embeddings_from_file(
            args.embeddings_path, verbose=True)

    if args.gpu:
        model.cuda()

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    # reduce rate by gamma every n epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.75)
    scheduler.verbose = True
    # kl annealing
    kl_schedule = kl_sigmoid_annealing_schedule(inflection=args.inflection)

    class VAETrainer(Trainer):
        def on_batch_end(self, epoch, batch, loss):
            # reset kl weight
            total_batches = len(self.datasets['train'])
            self.model.encoder.kl_weight = kl_schedule(
                batch + total_batches * epoch)

    losses = [{'loss': 'rec'},
              {'loss': 'kl', 'format': lambda loss: loss}]

    trainer = VAETrainer(
        model, {'train': train, 'valid': valid, 'test': test}, optimizer,
        losses=losses, early_stopping=EarlyStopping(args.patience),
        max_norm=args.max_norm, scheduler=scheduler)

    # hooks
    trainer.add_hook(make_generate_hook(args.level),
                     hooks_per_epoch=args.hooks_per_epoch)
    trainer.add_hook(kl_weight_hook, hooks_per_epoch=100)
    if args.use_schedule:
        hook = u.make_schedule_hook(
            inflection_sigmoid(len(train) * 2, 1.75, inverse=True))
        trainer.add_hook(hook, hooks_per_epoch=1000)

    trainer.add_loggers(
        StdLogger(),
        # VisdomLogger(env='vae', losses=('rec', 'kl'), max_y=600)
    )

    trainer.train(args.epochs, args.checkpoints,
                  shuffle=True, use_schedule=args.use_schedule)
