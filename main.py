
import string
import random
import utils as u

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

from encoder_decoder import EncoderDecoder

random.seed(1001)

train_len = 30000
val_len = 100
min_input_len = 5
max_input_len = 15
epochs = 30
bs = 100

vocab = list(string.ascii_letters)

vocab.append(u.EOS)
vocab.append(u.PAD)

int2char = list(vocab)
char2int = {c: i for i, c in enumerate(vocab)}

num_layers = (1, 1)
enc_num_layers, dec_num_layers = num_layers
emb_dim = 4
hid_dim = (124, 124)
enc_hid_dim, dec_hid_dim = hid_dim
att_dim = 64

sample_fn = u.reverse
train_set = u.generate_set(
    train_len, vocab, sample_fn=sample_fn,
    min_len=min_input_len, max_len=max_input_len)
val_set = u.generate_set(
    val_len, vocab,
    sample_fn=sample_fn,
    min_len=min_input_len, max_len=max_input_len)


def prepare_data(data, char2int):
    EOS = char2int[u.EOS]
    src, tgt = zip(*list(data))
    src = [[char2int[x] for x in seq] + [EOS] for seq in src]
    tgt = [[EOS] + [char2int[x] for x in seq] for seq in tgt]
    return src, tgt

train_src, train_tgt = prepare_data(train_set, char2int)
train_data = u.Dataset(train_src, train_tgt, bs, pad=char2int[u.PAD])

model = EncoderDecoder(
    num_layers, emb_dim, hid_dim, att_dim, char2int, int2char,
    dropout=0.0, add_prev=True, project_init=True)


def grad_norm(model):
    import math
    norm = 0
    for p in model.parameters():
        norm += math.pow(p.grad.data.norm(), 2)
    return norm


def init_params(model):
    for p in model.parameters():
        p.data.uniform_(-0.05, 0.05)


def train(model, train_data, epochs, char2int, pad,
          do_val=False, checkpoint=50):
    model.train()
    loss_weight = torch.ones(len(char2int))
    loss_weight[char2int[pad]] = 0  # don't penalize padding
    criterion = nn.NLLLoss(weight=loss_weight)
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    for e in range(epochs):
        epoch_loss = 0
        batch_order = torch.randperm(len(train_data))
        for num_batch, b in enumerate(batch_order):
            model.zero_grad()   # zeroes gradient buffers
            src, tgt = train_data[b]
            tgt = tgt[1:]       # remove <EOS>
            outs, (h_n, c_n) = model(src, tgt)
            logs = model.project(outs.view(-1, outs.size(2)))
            loss = criterion(logs, tgt.view(-1))
            loss.div(src.size(1)).backward()  # average over batch
            optimizer.step()
            epoch_loss += loss.data[0]
            assert(isinstance(loss.data[0], float))
            if num_batch > 0 and num_batch % checkpoint == 0:
                print("epoch/batch [%d:%d] | loss [%f]" %
                      (e, num_batch, epoch_loss/num_batch))

n_params = sum([p.nelement() for p in model.parameters()])
print("* number of parameter s: %d" % n_params)
train(model, train_data, epochs, char2int, u.PAD)

# src, tgt = train_data[0]
# out, _ = model(src, tgt)
# probs = model.project(out.view(-1, out.size(2)))
# loss_weight = torch.ones(len(char2int))
# loss_weight[char2int[u.PAD]] = 0  # don't penalize padding
# criterion = nn.NLLLoss(weight=loss_weight)
# optimizer = optim.Adadelta(model.parameters(), lr=0.01)
# loss = criterion(probs, tgt.view(-1))
# var = loss.creator
# while True:
#     var = var.previous_functions[0][0]
#     print(var)
