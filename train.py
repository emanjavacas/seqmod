
import time
import string
import math

seed = 1005

import random
random.seed(seed)

import torch
from torch import nn
from torch.autograd import Variable

try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

from hinton_diagram import hinton
from encoder_decoder import EncoderDecoder
from optimizer import Optimizer

from dataset import batchify
import dummy as d
import utils as u


def plot_weights(att_weights, target, pred, e, batch):
    fig = hinton(att_weights.squeeze(1).t().data.cpu().numpy(),
                 ylabels=list(target),
                 xlabels=list(pred.replace(u.EOS, '')))
    fig.savefig('./img/%d_%d' % (e, batch))


def translate(model, target, src_dict, gpu, beam):
    target = torch.LongTensor(list(src_dict.transform([target], bos=False))).t()
    batch = Variable(target, volatile=True)
    batch = batch.cuda() if gpu else batch
    if beam:
        preds, _ = model.translate_beam(
            batch, beam_width=5, max_decode_len=4)
    else:
        preds, _ = model.translate(batch, max_decode_len=4)
    return preds, None


def run_translation(model, target, src_dict, e, b, plot_att, gpu, beam):
    if target:
        i2s = {i: c for c, i in model.src_dict.items()}
        preds, att = translate(model, target, src_dict, gpu, beam)
        print("* " + ' '.join(target) if isinstance(target, list) else target)
        for idx, hyp in enumerate(preds):
            print("* [%d]: %s" % (idx, ' '.join(i2s[w] for w in hyp)))
        if plot_att:
            plot_weights(att, target, pred, e, b)


def validate_model(model, criterion, val_data, src_dict, e,
                   target=None, gpu=False, plot_att=False, beam=True):
    pad, eos = model.src_dict[u.PAD], model.src_dict[u.EOS]
    total_loss, total_words = 0, 0
    model.eval()
    for b in range(len(val_data)):
        batch = val_data[b]
        source, targets = batch
        # remove <eos> from decoder targets
        decode_targets = Variable(u.map_index(targets[:-1].data, eos, pad))
        # remove <bos> from sources
        outs = model(source[1:], decode_targets)
        # remove <bos> from loss targets
        loss, _ = batch_loss(model, outs, targets[1:], criterion, do_val=True)
        total_loss += loss
        total_words += targets.data.ne(pad).sum()
    run_translation(model, target, src_dict, e, b, plot_att, gpu, beam)
    return total_loss / total_words


def batch_loss(model, outs, targets, criterion, do_val=False, split_batch=52):
    loss = 0
    outs = Variable(outs.data, requires_grad=(not do_val), volatile=do_val)
    outs_split = torch.split(outs, split_batch)
    targets_split = torch.split(targets, split_batch)
    for out, trg in zip(outs_split, targets_split):
        out = out.view(-1, out.size(2))
        loss += criterion(model.project(out), trg.view(-1))
    if not do_val:
        loss.div(outs.size(1)).backward()
    grad_output = None if outs.grad is None else outs.grad.data
    return loss.data[0], grad_output


def train_epoch(model, epoch, train_data, criterion, optimizer, checkpoint):
    start = time.time()
    epoch_loss, report_loss = 0, 0
    epoch_words, report_words = 0, 0
    pad, eos = model.src_dict[u.PAD], model.src_dict[u.EOS]
    batch_order = torch.randperm(len(train_data))

    for idx in range(len(train_data)):
        batch = train_data[batch_order[idx]]
        optimizer.optim.zero_grad()  # empty gradients at begin of batch
        source, targets = batch
        # remove <eos> from decoder targets
        decode_targets = Variable(u.map_index(targets[:-1].data, eos, pad))
        # remove <bos> from source
        outs = model(source[1:], decode_targets)
        # remove <bos> from loss targets
        loss, grad_output = batch_loss(model, outs, targets[1:], criterion)
        outs.backward(grad_output)
        optimizer.step()
        # report
        num_words = targets.data.ne(model.src_dict[u.PAD]).sum()
        epoch_words += num_words
        report_words += num_words
        epoch_loss += loss
        report_loss += loss
        if idx % checkpoint == 0 and idx > 0:
            print("Epoch %d, %5d/%5d batches; ppl: %6.2f; %3.0f tokens/s" %
                  (epoch, idx, len(train_data),
                   math.exp(report_loss / report_words),
                   report_words/(time.time()-start)))
            report_loss = report_words = 0
            start = time.time()

    return epoch_loss / epoch_words


def make_criterion(vocab_size, pad):
    weight = torch.ones(vocab_size)
    weight[pad] = 0
    criterion = nn.NLLLoss(weight, size_average=False)
    return criterion


def train_model(model, train_data, valid_data, optimizer, src_dict, epochs,
                checkpoint=50, gpu=False, target=None, beam=False):
    vocab_size = len(src_dict)
    criterion = make_criterion(vocab_size, src_dict.get_pad())

    if gpu:
        criterion.cuda()
        model.cuda()

    for epoch in range(1, epochs + 1):
        model.train()
        # train for one epoch on the training set
        train_loss = train_epoch(
            model, epoch, train_data, criterion, optimizer, checkpoint)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))
        # evaluate on the validation set
        val_loss = validate_model(
            model, criterion, valid_data, src_dict, epoch,
            gpu=gpu, target=target, beam=beam)
        val_ppl = math.exp(min(val_loss, 100))
        print('Validation perplexity: %g' % val_ppl)
        # maybe update the learning rate
        lr_data = optimizer.maybe_update_lr(epoch, val_loss)
        if lr_data is not None:
            print("Decayed learning rate [%f -> %f]" %
                  (lr_data['last_lr'], lr_data['new_lr']))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_len', default=10000, type=int)
    parser.add_argument('--target', default='redrum', type=str)
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--min_len', default=1, type=int)
    parser.add_argument('--max_len', default=15, type=int)
    parser.add_argument('--sample_fn', default='reverse', type=str)
    parser.add_argument('--bidi', action='store_true')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--emb_dim', default=4, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--att_dim', default=64, type=int)
    parser.add_argument('--att_type', default='Bahdanau', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--prefix', default='model', type=str)
    parser.add_argument('--vocab', default=list(string.ascii_letters) + [' '])
    parser.add_argument('--checkpoint', default=500, type=int)
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--learning_rate', default=1., type=float)
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

    train, val, src_dict = d.load_dummy_data(
        size, vocab, batch_size, min_len=args.min_len, max_len=args.max_len,
        sample_fn=getattr(d, args.sample_fn), gpu=args.gpu, dev=args.val_split)
    s2i = train.dataset.dicts['src'].s2i

    print(' * vocabulary size. %d' % len(src_dict))
    print(' * number of train batches. %d' % len(train))
    print(' * maximum batch size. %d' % batch_size)

    print('Building model...')

    model = EncoderDecoder(
        (args.layers, args.layers), args.emb_dim, (args.hid_dim, args.hid_dim),
        args.att_dim, s2i, att_type=args.att_type, dropout=args.dropout,
        bidi=args.bidi)

    model.apply(u.Initializer.make_initializer())
    # model.apply(u.default_weight_init)
    # model.init_params()

    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    print(model)

    train_model(model, train, val, optimizer, src_dict, args.epochs,
                gpu=args.gpu, target=list(args.target), beam=args.beam)
