
import time
import string
import math

import torch
from torch import nn
from torch.autograd import Variable

import utils as u
from encoder_decoder import EncoderDecoder
from optimizer import Optimizer


def make_criterion(vocab_size, pad):
    weight = torch.ones(vocab_size)
    weight[pad] = 0
    criterion = nn.NLLLoss(weight, size_average=False)
    return criterion


def batch_loss(model, outs, targets, criterion, do_val=False, split_batch=52):
    """
    compute generations one piece at a time
    """
    batch_loss = 0
    outs = Variable(outs.data, requires_grad=(not do_val), volatile=do_val)
    batch_size = outs.size(1)
    outs_split = torch.split(outs, split_batch)
    targets_split = torch.split(targets, split_batch)
    for out, targ in zip(outs_split, targets_split):
        out = out.view(-1, out.size(2))
        logs = model.project(out)
        loss = criterion(logs, targ.view(-1))
        batch_loss += loss.data[0]
        if not do_val:
            loss.div(batch_size).backward()

    grad_output = None if outs.grad is None else outs.grad.data
    return batch_loss, grad_output


def validate_model(model, criterion, val_data, pad):
    total_loss, total_words = 0, 0
    model.eval()
    for b in range(len(val_data)):
        source, targets = val_data[b]
        outs, _ = model(source, targets[:-1])
        targets = targets[1:]
        loss, _ = batch_loss(model, outs, targets, criterion, do_val=True)
        total_loss += loss
        total_words += targets.data.ne(pad).sum()
    model.train()
    return total_loss / total_words


def train_epoch(epoch, train_data, criterion, checkpoint):
    start = time.time()
    epoch_loss, report_loss = 0, 0
    epoch_words, report_words = 0, 0
    batch_order = torch.randperm(len(train_data))

    for b, idx in enumerate(batch_order):
        source, targets = train_data[idx]
        model.zero_grad()       # empty gradients at begin of batch
        outs, _ = model(source, targets[:-1])  # exclude last <eos> from train
        targets = targets[1:]   # exclude initial <eos> from targets from test
        loss, grad_output = batch_loss(model, outs, targets, criterion)
        outs.backward(grad_output)
        optim.step()

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


def train_model(model, train_data, valid_data, src_dict, optim, epochs,
                init_range=0.05, checkpoint=50, gpu=False):
    pad = char2int[u.PAD]
    criterion = make_criterion(len(src_dict), pad)

    if gpu:
        criterion.cuda()
        model.cuda()

    model.init_params(init_range=init_range)
    model.train()

    for epoch in range(1, epochs + 1):
        # train for one epoch on the training set
        train_loss = train_epoch(epoch, train_data, criterion, checkpoint)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))
        # evaluate on the validation set
        val_loss = validate_model(model, criterion, valid_data, pad)
        val_ppl = math.exp(min(val_loss, 100))
        print('Validation perplexity: %g' % val_ppl)
        # maybe update the learning rate
        if optim == 'SGD':
            optim.update_learning_rate(val_loss, epoch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_len', default=10000, type=int)
    parser.add_argument('-v', '--val_len', default=1000, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-m', '--min_input_len', default=1, type=int)
    parser.add_argument('-M', '--max_input_len', default=15, type=int)
    parser.add_argument('-f', '--sample_fn', default='reverse', type=str)
    parser.add_argument('-l', '--layers', default=1, type=int)
    parser.add_argument('-e', '--emb_dim', default=4, type=int)
    parser.add_argument('-H', '--hid_dim', default=64, type=int)
    parser.add_argument('-a', '--att_dim', default=64, type=int)
    parser.add_argument('-A', '--att_type', default='Bahdanau', type=str)
    parser.add_argument('-E', '--epochs', default=5, type=int)
    parser.add_argument('-p', '--prefix', default='model', type=str)
    parser.add_argument('-V', '--vocab', default=list(string.ascii_letters))
    parser.add_argument('-c', '--checkpoint', default=500, type=int)
    parser.add_argument('-o', '--optim', default='SGD', type=str)
    parser.add_argument('-P', '--plot', action='store_true')
    parser.add_argument('-r', '--learning_rate', default=1., type=float)
    parser.add_argument('-D', '--dropout', default=0.0, type=float)
    parser.add_argument('-d', '--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('-s', '--start_decay_at', default=8, type=int)
    parser.add_argument('-g', '--max_grad_norm', default=5., type=float)
    parser.add_argument('-G', '--gpu', action='store_true')
    args = parser.parse_args()

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
        args.att_dim, char2int, att_type=args.att_type, dropout=args.dropout)
    optim = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    print(model)
    train_model(
        model, train_data, val_data, char2int, optim, args.epochs,
        gpu=args.gpu)
