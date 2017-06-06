
import os
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from seqmod.modules.custom import StackedRNN
from seqmod.loaders import load_penn3
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.dataset import PairedDataset, default_sort_key
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.trainer import Trainer
from seqmod.misc.loggers import StdLogger


class POSAwareLM(nn.Module):
    def __init__(self, vocabs, emb_dims, hid_dims,
                 num_layers=(1, 1), dropout=0.0, cell='LSTM'):
        self.pos_vocab, self.word_vocab = vocabs
        self.pos_emb_dim, self.word_emb_dim = emb_dims
        self.pos_hid_dim, self.word_hid_dim = hid_dims
        self.pos_num_layers, self.word_num_layers = num_layers
        self.cell = cell
        super(POSAwareLM, self).__init__()

        # pos network
        self.pos_emb = nn.Embedding(self.pos_vocab, self.pos_emb_dim)
        self.pos_rnn = StackedRNN(
            self.pos_num_layers,
            self.pos_emb_dim + self.word_emb_dim,
            self.pos_hid_dim,
            cell=cell)
        self.pos_project = nn.Sequential(
            nn.Linear(self.pos_hid_dim, self.pos_vocab),
            nn.LogSoftmax())

        # word network
        self.word_emb = nn.Embedding(self.word_vocab, self.word_emb_dim)
        self.word_rnn = StackedRNN(
            self.word_num_layers,
            self.word_emb_dim + self.pos_emb_dim,
            self.word_hid_dim,
            cell=cell)
        self.word_project = nn.Sequential(
            nn.Linear(self.word_hid_dim, self.word_vocab),
            nn.LogSoftmax())

    def init_hidden_for(self, inp, source):
        batch = inp.size(0)
        if source == 'pos':
            size = (self.pos_num_layers, batch, self.pos_hid_dim)
        else:
            assert source == 'word'
            size = (self.word_num_layers, batch, self.word_hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def forward(self, pos, word):
        """
        <bos>/<bos> NNP/Pierre NNP/Vinken CD/61 ... ,/, MD/will <eos>/<eos>

        ==Input==
              1 (pos/word) 2          3                n-1
        POS:  <bos>/<bos>  NNP/Pierre NNP/Vinken  ...  MD/will
        word: NNP/<bos>    NNP/Pierre CD/Vinken   ...  <eos>/will

        ==Output==
              1            2          3                n-1
        POS:  NNP          NNP        CD          ...  <eos>
        word: Pierre       Vinken     61          ...  <eos>
        """
        pos_outs, word_outs, p_hid, w_hid = [], [], None, None
        pos_emb, word_emb = self.pos_emb(pos), self.word_emb(word)
        for idx in range(len(pos) - 1):
            pos_inp = torch.cat((pos_emb[idx], word_emb[idx]), 1)
            p_out, p_hid = self.pos_rnn(
                pos_inp, p_hid or self.init_hidden_for(pos_inp, 'pos'))
            word_inp = torch.cat((pos_emb[idx + 1], word_emb[idx]), 1)
            w_out, w_hid = self.word_rnn(
                word_inp, w_hid or self.init_hidden_for(word_inp, 'word'))
            pos_outs.append(self.pos_project(p_out))
            word_outs.append(self.word_project(w_out))
        return torch.stack(pos_outs), torch.stack(word_outs)

    def generate(self, pos_dict, word_dict,
                 temperature=1, seed=None, max_len_seq=25, gpu=False):
        p_hyp, w_hyp, p_hid, w_hid = [], [], None, None
        p_scores, w_scores = [], []
        word_eos = word_dict.get_eos()
        pos_prev = Variable(
            torch.LongTensor([pos_dict.get_bos()]),
            volatile=True)
        word_prev = Variable(
            torch.LongTensor([word_dict.get_bos()]),
            volatile=True)
        if gpu:
            pos_prev, word_prev = pos_prev.cuda(), word_prev.cuda()
        while (len(w_hyp) < 25 and
               word_prev.data.eq(word_eos).nonzero().nelement() == 0):
            # pos
            pos_emb = self.pos_emb(pos_prev)
            word_emb = self.word_emb(word_prev)
            p_inp = torch.cat((pos_emb, word_emb), 1)
            p_out, p_hid = self.pos_rnn(
                p_inp, p_hid or self.init_hidden_for(p_inp, 'pos'))
            pos_prev = p_out.div(temperature).exp_().multinomial().squeeze(0)
            p_hyp.append(pos_prev.data[0])
            p_score = p_out.squeeze()[p_hyp[-1]]
            p_scores.append(p_score.squeeze().data[0])
            # word
            pos_emb = self.pos_emb(pos_prev)
            w_inp = torch.cat((pos_emb, word_emb), 1)
            w_out, w_hid = self.word_rnn(
                w_inp, w_hid or self.init_hidden_for(w_inp, 'word'))
            word_prev = w_out.div(temperature).exp_().multinomial().squeeze(0)
            w_hyp.append(word_prev.data[0])
            w_score = w_out.squeeze()[w_hyp[-1]]
            w_scores.append(w_score.squeeze().data[0])

        p_score = sum(p_scores) / len(p_scores)
        w_score = sum(w_scores) / len(w_scores)
        return p_hyp, w_hyp, p_score, w_score


class POSAwareLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(POSAwareLMTrainer, self).__init__(*args, **kwargs)
        self.loss_labels = ('pos', 'word')
        self.size_average = False

    def format_loss(self, losses):
        return tuple(math.exp(min(loss, 100)) for loss in losses)

    def run_batch(self, batch_data, dataset='train', **kwargs):
        (src_pos, src_word), _ = batch_data
        pos_out, word_out = self.model(src_pos, src_word)
        seqlen, batch, _ = pos_out.size()
        pos_loss, word_loss = self.criterion(
            pos_out.view(seqlen * batch, -1), src_pos[1:].view(-1),
            word_out.view(seqlen * batch, -1), src_word[1:].view(-1))
        if dataset == 'train':
            (pos_loss + word_loss).div(batch).backward()
            self.optimizer_step()
        return pos_loss.data[0], word_loss.data[0]

    def num_batch_examples(self, batch_data):
        (pos_src, _), _ = batch_data
        pos_dict, _ = self.datasets['train'].d['src']
        return pos_src.data.ne(pos_dict.get_pad()).sum()


def make_pos_word_criterion(pos_vocab, pos_pad, word_vocab, word_pad,
                            gpu=False):

    def make_criterion(vocab, pad):
        weight = torch.ones(vocab)
        weight[pad] = 0
        crit = nn.NLLLoss(weight, size_average=False)
        if gpu:
            crit.cuda()
        return crit

    pos_crit = make_criterion(pos_vocab, pos_pad)
    word_crit = make_criterion(word_vocab, word_pad)

    def criterion(pos_logs, pos_targets, word_logs, word_targets):
        return (pos_crit(pos_logs, pos_targets),
                word_crit(word_logs, word_targets))

    return criterion


def make_generate_hook(pos_dict, word_dict):

    def hook(trainer, epoch, batch, checkpoints):
        p_hyp, w_hyp, p_score, w_score = \
            trainer.model.generate(pos_dict, word_dict)
        p_str, w_str = "", ""
        for p, w in zip(p_hyp, w_hyp):
            p = pos_dict.vocab[p]
            w = word_dict.vocab[w]
            ljust = max(len(p), len(w)) + 2
            p_str += p.ljust(ljust, ' ')
            w_str += w.ljust(ljust, ' ')
        trainer.log("info", "Score [%g, %g]: \n%s\n%s" %
                    (p_score, w_score, w_str, p_str))

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', default='/home/enrique/corpora/penn3/')
    parser.add_argument('--dataset_path')
    parser.add_argument('--load_dataset', action='store_true')
    parser.add_argument('--save_dataset', action='store_true')
    # model
    parser.add_argument('--pos_emb_dim', default=24, type=int)
    parser.add_argument('--word_emb_dim', default=64, type=int)
    parser.add_argument('--pos_hid_dim', default=100, type=int)
    parser.add_argument('--word_hid_dim', default=200, type=int)
    parser.add_argument('--pos_num_layers', default=1, type=int)
    parser.add_argument('--word_num_layers', default=1, type=int)
    # train
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--num_checkpoints', default=10, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    if args.load_dataset:
        dataset = PairedDataset.from_disk(args.dataset_path)
    else:
        words, pos = zip(*load_penn3(args.path, swbd=False))
        word_dict = Dict(
            pad_token='<pad>', eos_token='<eos>', bos_token='<bos>',
            force_unk=True, max_size=1000000)
        pos_dict = Dict(
            pad_token='<pad>', eos_token='<eos>', bos_token='<bos>',
            force_unk=False)
        word_dict.fit(words)
        pos_dict.fit(pos)
        dataset = PairedDataset(
            (pos, words), None, {'src': (pos_dict, word_dict)},
            batch_size=args.batch_size)
        # dataset = BlockDataset(
        #     (pos, words), (pos_dict, word_dict), args.batch_size, 20)
        if args.save_dataset and not os.path.isfile(args.dataset_path):
            dataset.to_disk(args.dataset_path)
    train, valid = dataset.splits(
        test=None,
        sort_key=default_sort_key, shuffle=True)

    pos_dict, word_dict = train.d['src']

    m = POSAwareLM(
        (len(pos_dict.vocab), len(word_dict.vocab)),  # vocabs
        (args.pos_emb_dim, args.word_emb_dim),
        (args.pos_hid_dim, args.word_hid_dim),
        num_layers=(args.pos_num_layers, args.word_num_layers), dropout=0.3)

    if args.gpu:
        m.cuda(), train.set_gpu(args.gpu), valid.set_gpu(args.gpu)

    crit = make_pos_word_criterion(
        len(pos_dict.vocab), pos_dict.get_pad(),
        len(word_dict.vocab), word_dict.get_pad(),
        gpu=args.gpu)
    optim = Optimizer(m.parameters(), 'Adam', lr=0.001)
    trainer = POSAwareLMTrainer(
        m, {'train': train, 'valid': valid}, crit, optim)
    trainer.add_loggers(StdLogger())
    trainer.add_hook(
        make_generate_hook(pos_dict, word_dict), num_checkpoints=5)
    trainer.train(args.epochs, args.num_checkpoints, suffle=True)
