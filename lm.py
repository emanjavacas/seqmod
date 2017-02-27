

import time
import math

seed = 1001
import random
random.seed(seed)

import torch
try:
    torch.cuda.manual_seed(seed)
except:
    print('no NVIDIA driver found')
torch.manual_seed(seed)

import torch.nn as nn
from torch.autograd import Variable

from optimizer import Optimizer
from dataset import Dict
from preprocess import text_processor
import utils as u


class LM(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, num_layers=1, cell='LSTM', bias=True):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        
        super(LM, self).__init__()
        self.embeddings = nn.Embedding(vocab, emb_dim)
        self.rnn = getattr(nn, cell)(emb_dim, hid_dim, num_layers, bias=bias)
        self.project = nn.Linear(hid_dim, vocab)

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('GRU'):
            return h_0
        else:
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0

    def forward(self, inp, hidden=None):
        emb = self.embeddings(inp)
        hidden = hidden or self.init_hidden_for(emb)
        out, hidden = self.rnn(emb, hidden)
        seq_len, batch, hid = out.size()
        # (seq_len x batch x hid) -> (seq_len * batch x hid)
        logs = self.project(out.view(seq_len * batch, hid))
        return logs, hidden


# Load data
def load_tokens(path, processor=text_processor()):
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if processor is not None:
                line = processor(line)
            line = line.split()
            if line:
                tokens.extend(line)
    return tokens


def load_from_file(path):
    if path.endswith('npy'):
        import numpy as np
        data = torch.LongTensor(np.load(path))
    elif path.endswith('.pt'):
        data = torch.load(path)
    else:
        raise ValueError('Unknown input format [%s]' % path)
    return data


def load_dict(path):
    if path.endswith('pickle'):
        import pickle as p
        with open(path, 'rb') as f:
            return p.load(f)
    elif path.endswith('pt'):
        return torch.load(path)
    else:
        raise ValueError('Unknown input format [%s]' % path)


def batchify(data, batch_size, gpu=False):
    num_batches = len(data) // batch_size
    data = data.narrow(0, 0, num_batches * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    if gpu:
        data = data.cuda()
    return data


def get_batch(data, i, bptt, evaluation=False, gpu=False):
    seq_len = min(bptt, len(data) - 1 - i)
    src = Variable(data[i:i+seq_len], volatile=evaluation)
    trg = Variable(data[i+1:i+seq_len+1].view(-1))
    if gpu:
        src, trg = src.cuda(), trg.cuda()
    return src, trg


# Training code
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def validate_model(model, data, bptt, criterion, gpu):
    loss, hidden = 0, None
    for i in range(0, data.size(0) - 1, bptt):
        source, targets = get_batch(data, i, bptt, evaluation=True, gpu=gpu)
        output, hidden = model(source, hidden=hidden)
        loss += len(source) * criterion(output, targets).data[0]
        hidden = repackage_hidden(hidden)
    return loss / len(data)


def train_epoch(model, data, optimizer, criterion, checkpoint, epoch, bptt, gpu):
    epoch_loss, batch_loss, report_words = 0, 0, 0
    start = time.time()
    hidden = None

    for batch, i in enumerate(range(0, len(data) - 1, bptt)):
        model.zero_grad()
        source, targets = get_batch(data, i, bptt, gpu=gpu)
        output, hidden = model(source, hidden)
        hidden = repackage_hidden(hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += len(source) * loss.data[0]
        batch_loss += loss.data[0]
        report_words += targets.nelement()

        if batch % checkpoint == 0 and batch > 0:
            print("Epoch %d, %5d/%5d batches; ppl: %6.2f; %3.0f tokens/s" %
                  (epoch, batch, len(data) // bptt,
                   math.exp(batch_loss / checkpoint),
                   report_words / (time.time() - start)))
            report_words = batch_loss = 0
            start = time.time()
    return epoch_loss / len(data)


def train_model(model, train, valid, test, optimizer, epochs, bptt,
                checkpoint=50, gpu=False, criterion=nn.CrossEntropyLoss()):
    if gpu:
        criterion.cuda()
        model.cuda()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_epoch(
            model, train, optimizer, criterion, checkpoint, epoch, bptt, gpu)
        print("Train perplexity: %g" % math.exp(min(train_loss, 100)))
        model.eval()
        valid_loss = validate_model(model, valid, bptt, criterion, gpu)
        print("Valid perplexity: %g" % math.exp(min(valid_loss, 100)))
    test_loss = validate_model(model, test, bptt, criterion, gpu)
    print("Test perplexity: %g" % math(exp(test_loss)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--processed', action='store_true',
                        help='Is data in processed format?')
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=200, type=int)
    parser.add_argument('--hid_dim', default=200, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--bptt', default=20, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--optim', default='RMSprop', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save', default='', type=str)
    args = parser.parse_args()

    print("Loading datasets...")
    if args.processed:
        assert args.dict_path, "Processed data requires DICT_PATH"
        data, d = load_from_file(args.path), load_dict(args.dict_path)
        split = int(data.size(0) * 0.01)
        test_split, val_split = data.size(0) - split, data.size(0) - (2 * split)
        train = batchify(data[:val_split], args.batch_size)
        valid = batchify(data[val_split:test_split], args.batch_size)
        test = batchify(data[:test_split], args.batch_size)
        del data
    else:
        train_tokens = load_tokens(args.path + 'train.txt')
        valid_tokens = load_tokens(args.path + 'valid.txt')
        test_tokens = load_tokens(args.path + 'test.txt')
        d = Dict(max_size=args.max_size, min_freq=args.min_freq, sequential=False)
        d.fit(train_tokens, valid_tokens, test_tokens)
        train = batchify(torch.LongTensor(list(d.transform(train_tokens))),
                         args.batch_size, gpu=args.gpu)
        valid = batchify(torch.LongTensor(list(d.transform(valid_tokens))),
                         args.batch_size, gpu=args.gpu)
        test = batchify(torch.LongTensor(list(d.transform(test_tokens))),
                        args.batch_size, gpu=args.gpu)
        del train_tokens, valid_tokens, test_tokens

    print(' * vocabulary size. %d' % len(d))
    print(' * number of train batches. %d' % len(train))

    print('Building model...')
    model = LM(len(d), args.emb_dim, args.hid_dim,
               num_layers=args.layers, cell=args.cell)

    model.apply(u.Initializer.make_initializer(
        rnn={'type': 'orthogonal', 'args': {'gain': 1.0}}))

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    print(model)

    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)

    train_model(model, train, valid, test, optimizer, args.epochs, args.bptt,
                gpu=args.gpu, checkpoint=args.checkpoint)

    if args.save != '':
        print('Saving model')
        with open(args.save, 'wb') as f:
            torch.save(model, f)
