
import time
import string

seed = 1005

import random                   # nopep8
random.seed(seed)

import torch                    # nopep8
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import numpy as np              # nopep8
np.random.seed(seed)

from torch import nn            # nopep8
from torch.autograd import Variable  # nopep8

# from hinton_diagram import hinton  # nopep8
from encoder_decoder import EncoderDecoder  # nopep8
from optimizer import Optimizer             # nopep8

from trainer import EncoderDecoderTrainer   # nopep8
from loggers import StdLogger, VisdomLogger  # nopep8
from dataset import PairedDataset, Dict      # nopep8
import dummy as d                            # nopep8
import utils as u                            # nopep8


# def plot_weights(att_weights, target, hyp, epoch, batch):
#     fig = hinton(att_weights.squeeze(1).t().data.cpu().numpy(),
#                  ylabels=list(target),
#                  xlabels=list(hyp.replace(u.EOS, '')))
#     fig.savefig('./img/{epoch}_{batch}'.format(epoch=epoch, batch=batch))


def translate(model, target, gpu, beam=False):
    target = torch.LongTensor(list(model.src_dict.transform([target], bos=False)))
    batch = Variable(target.t(), volatile=True)
    batch = batch.cuda() if gpu else batch
    if beam:
        scores, hyps, att = model.translate_beam(
            batch, beam_width=5, max_decode_len=4)
    else:
        scores, hyps, att = model.translate(batch, max_decode_len=4)
    return scores, hyps, att


def make_encdec_hook(target, gpu, beam=False):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Translating %s" % target)
        scores, hyps, atts = translate(trainer.model, target, gpu, beam=beam)
        hyps = [u.format_hyp(sum(score), hyp, hyp_num + 1, trainer.model.trg_dict)
                for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
        trainer.log("info", '\n***' + ''.join(hyps) + '\n***')

    return hook


def make_att_hook(target, gpu, beam=False):
    assert not beam, "beam doesn't output attention yet"

    def hook(trainer, epoch, batch_num, checkpoint):
        scores, hyps, atts = translate(trainer.model, target, gpu, beam=beam)
        # grab first hyp (only one in translate modus)
        hyp = ' '.join([trainer.model.trg_dict.vocab[i] for i in hyps[0]])
        trainer.log("attention",
                    {"att": atts[0],
                     "score": sum(scores[0]) / len(hyps[0]),
                     "target": [trainer.model.trg_dict.bos_token] + list(target),
                     "hyp": hyp.split(),
                     "epoch": epoch,
                     "batch_num": batch_num})

    return hook
    


def make_criterion(vocab_size, pad):
    weight = torch.ones(vocab_size)
    weight[pad] = 0
    # don't average batches since num words is variable (depending on padding)
    criterion = nn.NLLLoss(weight, size_average=False)
    return criterion


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='', type=str)
    parser.add_argument('--train_len', default=10000, type=int)
    parser.add_argument('--target', default='redrum', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    parser.add_argument('--sample_fn', default='reverse', type=str)
    parser.add_argument('--dev', default=0.1, type=float)
    parser.add_argument('--bidi', action='store_true')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM', type=str)
    parser.add_argument('--emb_dim', default=4, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--att_dim', default=64, type=int)
    parser.add_argument('--att_type', default='Bahdanau', type=str)
    parser.add_argument('--maxout', default=0, type=int)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--project_on_tied_weights', action='store_true')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--prefix', default='model', type=str)
    parser.add_argument('--vocab', default=list(string.ascii_letters) + [' '])
    parser.add_argument('--checkpoint', default=100, type=int)
    parser.add_argument('--hooks_per_epoch', default=2, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--beam', action='store_true')
    args = parser.parse_args()

    vocab = args.vocab
    size = args.train_len
    batch_size = args.batch_size
    sample_fn = getattr(d, args.sample_fn)

    if args.path != '':
        with open(args.path, 'rb+') as f:
            dataset = PairedDataset.from_disk(f)
        dataset.set_batch_size(args.batch_size)
        dataset.set_gpu(args.gpu)
        train, valid = dataset.splits(
            sort_key=lambda pair: len(pair[0]), dev=args.dev, test=None)
        src_dict = dataset.dicts['src']
    else:
        src, trg = zip(*d.generate_set(
            size, vocab, args.min_len, args.max_len, sample_fn))
        src, trg = list(map(list, src)), list(map(list, trg))
        src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS)
        src_dict.fit(src, trg)
        train, valid = PairedDataset(
            src, trg, {'src': src_dict, 'trg': src_dict},
            batch_size=args.batch_size, gpu=args.gpu
        ).splits(dev=args.dev, test=None, sort_key=lambda pair: len(pair[0]))

    print(' * vocabulary size. %d' % len(src_dict))
    print(' * number of train batches. %d' % len(train))
    print(' * maximum batch size. %d' % batch_size)

    print('Building model...')

    model = EncoderDecoder(
        (args.layers, args.layers), args.emb_dim, (args.hid_dim, args.hid_dim),
        args.att_dim, src_dict, att_type=args.att_type, dropout=args.dropout,
        bidi=args.bidi, cell=args.cell, maxout=args.maxout,
        tie_weights=args.tie_weights,
        project_on_tied_weights=args.project_on_tied_weights)

    # model.freeze_submodule('encoder')
    # model.encoder.register_backward_hook(u.log_grad)
    # model.decoder.register_backward_hook(u.log_grad)

    model.apply(u.make_initializer(
        rnn={'type': 'orthogonal', 'args': {'gain': 1.0}}))

    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)

    criterion = make_criterion(len(src_dict), src_dict.get_pad())

    print('* number of parameters: %d' % model.n_params())
    print(model)

    if args.gpu:
        model.cuda(), criterion.cuda()

    trainer = EncoderDecoderTrainer(
        model, {'train': train, 'valid': valid}, criterion, optimizer)
    trainer.add_loggers(StdLogger(), VisdomLogger(env='encdec'))

    hook = make_encdec_hook(args.target, args.gpu)
    num_checkpoints = len(train) // (args.checkpoint * args.hooks_per_epoch)
    trainer.add_hook(hook, num_checkpoints=num_checkpoints)

    trainer.train(args.epochs, args.checkpoint, shuffle=True, gpu=args.gpu)
