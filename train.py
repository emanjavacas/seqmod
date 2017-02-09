
import time
import string
import math

import torch
from torch import nn
from torch.autograd import Variable

from hinton_diagram import hinton
from encoder_decoder import EncoderDecoder
from optimizer import Optimizer
import utils as u


def plot_weights(att_weights, target, pred, e, batch):
    fig = hinton(att_weights.squeeze(1).t().data.cpu().numpy(),
                 ylabels=list(target),
                 xlabels=list(pred.replace(u.EOS, '')))
    fig.savefig('./img/%d_%d' % (e, batch))


def translate(model, epoch, b, targets, pad, gpu=False):
    seqs = [[model.src_dict[c] for c in t] for t in targets]
    batch_data = u.batchify(seqs, pad)
    if gpu:
        batch = Variable(batch_data, volatile=True).cuda()
    else:
        batch = Variable(batch_data, volatile=True)
    # preds, att = model.translate(batch, max_decode_len=4)
    scores, preds = model.translate_beam(batch, beam_width=5, max_decode_len=4)
    return preds, None  # att


def visualize_targets(model, e, b, pad, targets, plot_target_id, gpu):
    if targets:
        int2char = {i: c for c, i in model.src_dict.items()}
        preds, att_weights = translate(model, e, b, targets, pad, gpu=gpu)
        for target, hyps in zip(targets, preds):
            print("* " + target)
            for idx, hyp in enumerate(hyps):
                print("* [%d]: %s" % (idx, ''.join(int2char[w] for w in hyp)))
        if plot_target_id:
            target, pred = targets[plot_target_id], preds[plot_target_id]
            plot_weights(att_weights, target, pred, e, b)


def validate_model(model, criterion, e, val_data, pad,
                   targets=None, gpu=False, plot_target_id=False):
    total_loss, total_words = 0, 0
    model.eval()
    for b in range(len(val_data)):
        src, tgt = val_data[b]
        outs, _, _ = model(src, tgt[:-1])
        tgt = tgt[1:]
        loss, _ = batch_loss(model, outs, tgt, criterion, do_val=True)
        total_loss += loss
        total_words += tgt.data.ne(pad).sum()
    visualize_targets(model, e, b, pad, targets, plot_target_id, gpu)
    return total_loss / total_words


def batch_loss(model, outs, targets, criterion, do_val=False, split_batch=52):
    """
    compute generations one piece at a time
    """
    total_loss = 0
    outs = Variable(outs.data, requires_grad=(not do_val), volatile=do_val)
    batch_size = outs.size(1)
    outs_split = torch.split(outs, split_batch)
    targets_split = torch.split(targets, split_batch)
    for out, targ in zip(outs_split, targets_split):
        out = out.view(-1, out.size(2))
        logs = model.project(out)
        loss = criterion(logs, targ.view(-1))
        total_loss += loss.data[0]
        if not do_val:
            loss.div(batch_size).backward()

    grad_output = None if outs.grad is None else outs.grad.data
    return total_loss, grad_output


def train_epoch(epoch, train_data, criterion, optimizer, checkpoint):
    start = time.time()
    epoch_loss, report_loss = 0, 0
    epoch_words, report_words = 0, 0
    batch_order = torch.randperm(len(train_data))

    for b, idx in enumerate(batch_order):
        optimizer.optim.zero_grad()   # empty gradients at begin of batch
        source, targets = train_data[idx]
        outs, _, _ = model(source, targets[:-1])  # exclude last <eos> at train
        targets = targets[1:]   # exclude initial <eos> from targets from test
        loss, grad_output = batch_loss(model, outs, targets, criterion)
        outs.backward(grad_output)
        optimizer.step()
        # report
        num_words = targets.data.ne(pad).sum()
        epoch_words += num_words
        report_words += num_words
        epoch_loss += loss
        report_loss += loss
        if b % checkpoint == 0 and b > 0:
            print("Epoch %d, %5d/%5d batches; ppl: %6.2f; %3.0f tokens/s" %
                  (epoch, b, len(train_data),
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


def train_model(model, train_data, valid_data, src_dict, optimizer, epochs,
                init_range=0.05, checkpoint=50, gpu=False, targets=None):
    pad = char2int[u.PAD]
    criterion = make_criterion(len(src_dict), pad)

    if gpu:
        criterion.cuda()
        model.cuda()

    model.init_params(init_range=init_range)

    for epoch in range(1, epochs + 1):
        model.train()
        # train for one epoch on the training set
        train_loss = train_epoch(
            epoch, train_data, criterion, optimizer, checkpoint)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))
        # evaluate on the validation set
        val_loss = validate_model(
            model, criterion, epoch, valid_data, pad, gpu=gpu, targets=targets)
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
    parser.add_argument('--targets', default=['redrum'], nargs='*')
    parser.add_argument('--val_len', default=1000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--min_input_len', default=1, type=int)
    parser.add_argument('--max_input_len', default=15, type=int)
    parser.add_argument('--sample_fn', default='reverse', type=str)
    parser.add_argument('--bidi', action='store_true')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--emb_dim', default=4, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--att_dim', default=64, type=int)
    parser.add_argument('--att_type', default='Bahdanau', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--prefix', default='model', type=str)
    parser.add_argument('--vocab', default=list(string.ascii_letters + ''))
    parser.add_argument('--checkpoint', default=500, type=int)
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--learning_rate', default=1., type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--seed', default=1003, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    vocab = args.vocab
    vocab.append(u.EOS)
    vocab.append(u.PAD)

    char2int = {c: i for i, c in enumerate(vocab)}
    pad, eos = char2int[u.PAD], char2int[u.EOS]

    # sample data
    train_set = u.generate_set(
        args.train_len, vocab, sample_fn=getattr(u, args.sample_fn),
        min_len=args.min_input_len, max_len=args.max_input_len)
    val_set = u.generate_set(
        args.val_len, vocab, sample_fn=getattr(u, args.sample_fn),
        min_len=args.min_input_len, max_len=args.max_input_len)
    train_data = u.prepare_data(
        train_set, char2int, args.batch_size, gpu=args.gpu)
    val_data = u.prepare_data(
        val_set, char2int, args.batch_size, gpu=args.gpu)

    print(' * vocabulary size. %d' % len(vocab))
    print(' * number of training sentences. %d' % len(train_data))
    print(' * maximum batch size. %d' % args.batch_size)

    print('Building model...')

    model = EncoderDecoder(
        (args.layers, args.layers), args.emb_dim, (args.hid_dim, args.hid_dim),
        args.att_dim, char2int, att_type=args.att_type, dropout=args.dropout,
        bidi=args.bidi)
    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    print(model)
    train_model(
        model, train_data, val_data, char2int, optimizer, args.epochs,
        gpu=args.gpu, targets=args.targets)
