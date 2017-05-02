
import os
import argparse

from loaders import load_twisty, load_dataset, load_embeddings
from vae import SequenceVAE
from vae_utils import vae_criterion, VAETrainer, kl_weight_hook
from seqmod import utils as u
from seqmod.misc.loggers import StdLogger, VisdomLogger
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.preprocess import text_processor
from seqmod.misc.dataset import PairedDataset, Dict


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


def make_mock_labels(lines, label='0'):
    return [[label] for _ in range(len(lines))]


def load_from_lines(
        path, batch_size, max_size=1000000, min_freq=5, gpu=False,
        shuffle=True, sort_key=lambda x: len(x[0]), **kwargs):
    lines = load_lines(path)
    ldict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
                 max_size=max_size, min_freq=min_freq)
    ldict.fit(lines)
    mock_labels = make_mock_labels(train)
    mock = Dict()
    mock.fit(mock_labels)
    d = {'src': ldict, 'trg': mock}
    splits = PairedDataset(lines, mock_labels, d, batch_size, gpu=gpu).splits(
        shuffle=shuffle, sort_key=sort_key, **kwargs)
    return splits


def load_penn(path, batch_size,
              max_size=1000000, min_freq=1, gpu=False, shuffle=True,
              sort_key=lambda pair: len(pair[0])):
    train_data = load_lines(os.path.join(path, 'train.txt'))
    train_labels = make_mock_labels(train_data)
    valid_data = load_lines(os.path.join(path, 'valid.txt'))
    valid_labels = make_mock_labels(valid_data)
    test_data = load_lines(os.path.join(path, 'test.txt'))
    test_labels = make_mock_labels(test_data)
    ldict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
                 max_size=max_size, min_freq=min_freq)
    ldict.fit(train_data, valid_data)
    mock = Dict().fit(train_labels)
    d = {'src': ldict, 'trg': mock}
    train = PairedDataset(train_data, train_labels, d, batch_size, gpu=gpu
    ).sort_(sort_key=sort_key)
    valid = PairedDataset(valid_data, valid_labels, d, batch_size, gpu=gpu,
                          evaluation=True).sort_(sort_key=sort_key)
    test = PairedDataset(test_data, test_labels, d, batch_size, gpu=gpu,
                         evaluation=True).sort_(sort_key=sort_key)
    return train, valid, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=0, type=int)
    parser.add_argument('--emb_dim', default=50, type=int)
    parser.add_argument('--hid_dim', default=50, type=int)
    parser.add_argument('--dec_hid_dim', default=0, type=int)
    parser.add_argument('--z_dim', default=50, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--non_bidi', action='store_true')
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_on_tied_weights', action='store_true')
    parser.add_argument('--project_init', action='store_true')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--add_z', action='store_true')
    parser.add_argument('--load_embeddings', action='store_true')
    parser.add_argument('--flavor', default=None)
    parser.add_argument('--suffix', default=None)
    # training
    parser.add_argument('--optim', default='RMSprop')
    parser.add_argument('--learning_rate', default=0.01, type=float)
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

    print("Loading data...")
    prefix = '{source}.{level}.{min_len}.{min_freq}.{concat}.{max_size}' \
             .format(**vars(args))
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
        if args.cache_data:
            train.to_disk('data/%s_train.pt' % prefix)
            test.to_disk('data/%s_test.pt' % prefix)
            valid.to_disk('data/%s_valid.pt' % prefix)
    else:
        train = PairedDataset.from_disk('data/%s_train.pt' % prefix)
        test = PairedDataset.from_disk('data/%s_test.pt' % prefix)
        valid = PairedDataset.from_disk('data/%s_valid.pt' % prefix)
        train.set_gpu(args.gpu), test.set_gpu(args.gpu), valid.set_gpu(args.gpu)
        train.set_batch_size(args.batch_size), test.set_batch_size(args.batch_size)
        valid.set_batch_size(args.batch_size)
    datasets = {'train': train, 'valid': valid, 'test': test}
    vocab = len(train.d['src'].vocab)
    print("* Number of train batches %d" % len(train))

    print("Building model...")
    hid_dim = args.hid_dim if args.dec_hid_dim == 0 else (args.hid_dim, args.dec_hid_dim)
    layers = args.layers if args.dec_layers == 0 else (args.layers, args.dec_layers)
    model = SequenceVAE(
        args.emb_dim, hid_dim, args.z_dim, train.d['src'],
        num_layers=layers, cell=args.cell, bidi=not args.non_bidi,
        dropout=args.dropout, add_z=args.add_z, word_dropout=args.word_dropout,
        tie_weights=args.tie_weights, project_init=args.project_init,
        project_on_tied_weights=args.project_on_tied_weights)
    print(model)
    model.apply(u.make_initializer())
    # model.encoder.register_backward_hook(u.log_grad)
    
    if args.load_embeddings:
        weight = load_embeddings(
            train.d['src'].vocab, args.flavor, args.suffix, '~/data/word_embeddings')
        model.init_embeddings(weight)

    criterion = vae_criterion(vocab, train.d['src'].get_pad())

    if args.gpu:
        model.cuda(), criterion.cuda()

    def on_lr_update(old_lr, new_lr):
        trainer.log("info", "Resetting lr [%g -> %g]" % (old_lr, new_lr))

    optimizer = Optimizer(
        model.parameters(), args.optim, lr=args.learning_rate,
        max_norm=args.max_norm, weight_decay=args.weight_decay,
        # SGD-only
        start_decay_at=args.start_decay_at, lr_decay=args.lr_decay,
        on_lr_update=on_lr_update)

    trainer = VAETrainer(
        model, datasets, criterion, optimizer, inflection_point=args.inflection_point)
    trainer.add_loggers(
        StdLogger(),
        VisdomLogger(env='vae_gender', losses=('rec', 'kl'), max_y=600))
    # trainer.add_hook(kl_weight_hook)

    trainer.train(args.epochs, args.checkpoints, shuffle=True, gpu=args.gpu)
