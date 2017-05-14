
import string

seed = 1001

import random
random.seed(seed)

import torch
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import dummy as d
from train_encoder_decoder import make_encdec_hook, make_att_hook
from train_encoder_decoder import make_criterion

from seqmod.modules.encoder_decoder import ForkableMultiTarget
from seqmod import utils as u

from seqmod.misc.dataset import PairedDataset, Dict
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.trainer import EncoderDecoderTrainer
from seqmod.misc.loggers import StdLogger, VisdomLogger


def wrap_autoencode(sample_fn):
    def wrapped(string):
        src, trg = sample_fn(string)
        return trg, trg
    return wrapped


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', type=str, required=True)
    parser.add_argument('--train_len', default=100000, type=int)
    parser.add_argument('--target', default='redrum', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--bidi', action='store_true')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM', type=str)
    parser.add_argument('--emb_dim', default=4, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--att_dim', default=64, type=int)
    parser.add_argument('--att_type', default='Bahdanau', type=str)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--project_init', action='store_true')
    parser.add_argument('--maxout', default=0, type=int)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--vocab', default=list(string.ascii_letters) + [' '])
    parser.add_argument('--checkpoint', default=100, type=int)
    parser.add_argument('--hooks_per_epoch', default=5, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--beam', action='store_true')
    args = parser.parse_args()

    datasets = {}
    for target in args.targets:
        sample_fn = wrap_autoencode(getattr(d, target))
        src, trg = zip(*d.generate_set(
            args.train_len, args.vocab, args.min_len, args.max_len, sample_fn))
        src, trg = list(map(list, src)), list(map(list, trg))
        datasets[target] = {'src': src, 'trg': trg}

    src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS)
    src_dict.fit(*[data
                   for target in datasets
                   for data in datasets[target].values()])

    for target in datasets:
        train, valid = PairedDataset(
            datasets[target]['src'], datasets[target]['trg'],
            {'src': src_dict, 'trg': src_dict},
            batch_size=args.batch_size, gpu=args.gpu).splits(
                dev=args.dev, test=None,
                shuffle=True, sort_key=lambda pair: len(pair[0]))
        del datasets[target]
        src, trg = zip(*d.generate_set(
            int(args.train_len * 0.1), args.vocab, args.min_len, args.max_len,
            getattr(d, target)))
        src, trg = list(map(list, src)), list(map(list, trg))
        test = PairedDataset(src, trg, {'src': src_dict, 'trg': src_dict},
                             batch_size=args.batch_size, gpu=args.gpu)
        datasets[target] = {'train': train, 'valid': valid, 'test': test}

    print('Building model...')
    model = ForkableMultiTarget(
        (args.layers, args.layers), args.emb_dim, (args.hid_dim, args.hid_dim),
        args.att_dim, src_dict, att_type=args.att_type, dropout=args.dropout,
        word_dropout=args.word_dropout, bidi=args.bidi, cell=args.cell,
        project_init=args.project_init)
    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)
    criterion = make_criterion(len(src_dict), src_dict.get_pad())

    model.apply(u.make_initializer(
        rnn={'type': 'orthogonal', 'args': {'gain': 1.0}}))

    if args.gpu:
        model.cuda(), criterion.cuda()

    hook = make_encdec_hook(args.target, args.gpu)

    # train general model
    train = PairedDataset(
        [s for t in datasets for s in datasets[t]['train'].data['src']],
        [s for t in datasets for s in datasets[t]['train'].data['trg']],
        {'src': src_dict, 'trg': src_dict},
        batch_size=args.batch_size, gpu=args.gpu,
        fitted=True)
    valid = PairedDataset(
        [s for t in datasets for s in datasets[t]['valid'].data['src']],
        [s for t in datasets for s in datasets[t]['valid'].data['trg']],
        {'src': src_dict, 'trg': src_dict},
        batch_size=args.batch_size, gpu=args.gpu,
        fitted=True)
    test = PairedDataset(
        [s for t in datasets for s in datasets[t]['test'].data['src']],
        [s for t in datasets for s in datasets[t]['test'].data['trg']],
        {'src': src_dict, 'trg': src_dict},
        batch_size=args.batch_size, gpu=args.gpu,
        fitted=True)

    stdlogger = StdLogger(outputfile='.multitarget.log')
    trainer = EncoderDecoderTrainer(
        model, {'train': train, 'valid': valid, 'test': test},
        criterion, optimizer)
    trainer.add_loggers(
        stdlogger, VisdomLogger(env='multitarget', title='general'))
    num_checkpoints = \
        max(len(train) // (args.checkpoint * args.hooks_per_epoch), 1)
    trainer.add_hook(hook, num_checkpoints=num_checkpoints)

    trainer.log('info', ' * vocabulary size. %d' % len(src_dict))
    trainer.log('info', ' * maximum batch size. %d' % args.batch_size)
    trainer.log('info', ' * number of train batches. %d' % len(train))
    trainer.log('info', ' * number of parameters. %d' % model.n_params())
    trainer.log('info', str(model))
    trainer.log('info', "**********************")
    trainer.log('info', "Training general model")
    trainer.train(args.epochs, args.checkpoint, shuffle=True, gpu=args.gpu)

    # train decoders
    for target in datasets:
        fork = model.fork_target(
            rnn={'type': 'orthogonal', 'args': {'gain': 1.0}})
        train = datasets[target]['train']
        trainer.log('info', '**********************')
        trainer.log('info', 'Training for target: %s' % target)
        trainer.log(
            'info', ' * number of unfrozen parameters: %d.' % fork.n_params())
        trainer.log('info', ' * number of train batches. %d' % len(train))
        trainer = EncoderDecoderTrainer(
            model, datasets[target], criterion, optimizer)
        trainer.add_loggers(
            stdlogger, VisdomLogger(env='multitarget', title=target))
        num_checkpoints = \
            max(1, len(train) // (args.checkpoint * args.hooks_per_epoch))
        trainer.add_hook(hook, num_checkpoints=num_checkpoints)
        trainer.add_hook(make_att_hook(args.target, args.gpu),
                         num_checkpoints=num_checkpoints)
        trainer.train(args.epochs, args.checkpoint, shuffle=True, gpu=args.gpu)

        # TODO: save model
