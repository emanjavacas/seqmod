
import os
import math

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from seqmod import utils as u

from seqmod.modules.custom import StackedRNN
from seqmod.loaders import load_penn3
from seqmod.misc.dataset import Dict, BlockDataset
from seqmod.misc.optimizer import Optimizer
from seqmod.misc.trainer import Trainer
from seqmod.misc.loggers import StdLogger


class POSAwareLM(nn.Module):
    def __init__(self, vocab, emb_dim, hid_dim, num_layers,
                 dropout=0.0, cell='LSTM', tie_weights=False):
        self.pos_vocab, self.word_vocab = vocab
        self.pos_emb_dim, self.word_emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        super(POSAwareLM, self).__init__()

        # embeddings
        self.pos_emb = nn.Embedding(self.pos_vocab, self.pos_emb_dim)
        self.word_emb = nn.Embedding(self.word_vocab, self.word_emb_dim)

    def init_hidden_for(self, inp, **kwargs):
        raise NotImplementedError

    def forward(self, pos, word, hidden=None):
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
        raise NotImplementedError

    def generate(self, pos_dict, word_dict,
                 seed=None, temperature=1, max_seq_len=25, gpu=False):
        raise NotImplementedError


class DoubleRNNPOSAwareLM(POSAwareLM):
    def __init__(self, *args, **kwargs):
        super(DoubleRNNPOSAwareLM, self).__init__(*args, **kwargs)
        if not isinstance(self.hid_dim, tuple):
            raise ValueError("two hid_dim params needed for double network")
        if not isinstance(self.num_layers, tuple):
            raise ValueError("two num_layers must be tuple for double network")
        self.pos_hid_dim, self.word_hid_dim = self.hid_dim
        self.pos_num_layers, self.word_num_layers = self.num_layers

        # pos network
        self.pos_rnn = StackedRNN(
            self.pos_num_layers,
            self.pos_emb_dim + self.word_hid_dim,
            self.pos_hid_dim,
            cell=self.cell)
        self.pos_project = nn.Sequential(
            nn.Linear(self.pos_hid_dim, self.pos_vocab),
            nn.LogSoftmax())

        # word network
        self.word_rnn = StackedRNN(
            self.word_num_layers,
            # TODO: add rnn output instead of argmax embedding
            self.word_emb_dim + self.pos_hid_dim,
            self.word_hid_dim,
            cell=self.cell)
        self.word_project = nn.Sequential(
            nn.Linear(self.word_hid_dim, self.word_vocab),
            nn.LogSoftmax())

    def init_hidden_for(self, inp, source_type):
        batch = inp.size(0)
        if source_type == 'pos':
            size = (self.pos_num_layers, batch, self.pos_hid_dim)
        else:
            assert source_type == 'word'
            size = (self.word_num_layers, batch, self.word_hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('LSTM'):
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0
        else:
            return h_0

    def get_last_hid(self, h):
        if self.cell.startswith('LSTM'):
            h, _ = h
        return h[-1]

    def forward(self, pos, word, hidden=None):
        p_outs, w_outs = [], []
        p_hid, w_hid = hidden if hidden is not None else (None, None)
        p_emb, w_emb = self.pos_emb(pos), self.word_emb(word)
        for p, w in zip(p_emb, w_emb):
            w_hid = w_hid or self.init_hidden_for(w, 'word')
            p_hid = p_hid or self.init_hidden_for(p, 'pos')
            p_out, p_hid = self.pos_rnn(
                torch.cat((p, self.get_last_hid(w_hid)), 1),
                p_hid)
            w_out, w_hid = self.word_rnn(
                torch.cat((w, self.get_last_hid(p_hid)), 1),
                w_hid)
            p_outs.append(self.pos_project(p_out))
            w_outs.append(self.word_project(w_out))
        return (torch.stack(p_outs), torch.stack(w_outs)), (p_hid, w_hid)

    def generate(self, p_dict, w_dict, seed=None, max_seq_len=20,
                 temperature=1., batch_size=1, gpu=False, ignore_eos=False):
        def sample(out):
            prev = out.div(temperature).exp_().multinomial().t()
            score = u.select_cols(out.data.cpu(), prev.squeeze().data.cpu())
            return prev, score

        p_hyp, w_hyp, p_hid, w_hid = [], [], None, None
        p_scores, w_scores = 0, 0
        w_eos = word_dict.get_eos()
        finished = np.array([False] * batch_size)
        p_prev = torch.LongTensor([pos_dict.get_bos()] * batch_size)
        w_prev = torch.LongTensor([word_dict.get_bos()] * batch_size)
        p_prev = Variable(p_prev, volatile=True)
        w_prev = Variable(w_prev, volatile=True)
        if gpu:
            p_prev, w_prev = p_prev.cuda(), w_prev.cuda()
        for _ in range(max_seq_len):
            # pos
            p_emb, w_emb = self.pos_emb(p_prev), self.word_emb(w_prev)
            w_hid = w_hid or self.init_hidden_for(w_emb, 'word')
            p_hid = p_hid or self.init_hidden_for(p_emb, 'pos')
            p_out, p_hid = self.pos_rnn(
                torch.cat((p_emb, self.get_last_hid(w_hid)), 1),
                p_hid)
            # word
            w_out, w_hid = self.word_rnn(
                torch.cat((w_emb, self.get_last_hid(p_hid)), 1),
                w_hid)
            (p_prev, p_score), (w_prev, w_score) = sample(p_out), sample(w_out)
            # hyps
            mask = (w_prev.squeeze().data == w_eos).cpu().numpy() == 1
            finished[mask] = True
            if all(finished == True):  # nopep8
                break
            p_hyp.append(p_prev.data[0]), w_hyp.append(w_prev.data[0])
            # scores
            p_score[torch.ByteTensor(finished.tolist())] = 0
            w_score[torch.ByteTensor(finished.tolist())] = 0
            p_scores, w_scores = p_scores + p_score, w_scores + w_score

        return (list(zip(*p_hyp)), list(zip(*w_hyp))), \
            (p_score.tolist(), w_score.tolist())


def repackage_hidden(hidden):
    def _repackage_hidden(h):
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(_repackage_hidden(v) for v in h)
    p_hid, w_hid = hidden
    return (_repackage_hidden(p_hid), _repackage_hidden(w_hid))


class POSAwareLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(POSAwareLMTrainer, self).__init__(*args, **kwargs)
        self.loss_labels = ('pos', 'word')

    def format_loss(self, losses):
        return tuple(math.exp(min(loss, 100)) for loss in losses)

    def run_batch(self, batch_data, dataset='train', **kwargs):
        (src_pos, src_word), (trg_pos, trg_word) = batch_data
        seq_len, batch_size = src_pos.size()
        hidden = self.batch_state.get('hidden')
        (p_out, w_out), hidden = self.model(src_pos, src_word, hidden)
        self.batch_state['hidden'] = repackage_hidden(hidden)
        p_loss, w_loss = self.criterion(
            p_out.view(seq_len * batch_size, -1), src_pos.view(-1),
            w_out.view(seq_len * batch_size, -1), src_word.view(-1))
        if dataset == 'train':
            (p_loss + w_loss).backward()
            self.optimizer_step()
        return p_loss.data[0], w_loss.data[0]

    def num_batch_examples(self, batch_data):
        (pos_src, _), _ = batch_data
        return pos_src.nelement()


def make_pos_word_criterion(gpu=False):
    p_crit, w_crit = nn.NLLLoss(), nn.NLLLoss()
    if gpu:
        p_crit.cuda(), w_crit.cuda()

    def criterion(p_outs, p_targets, w_outs, w_targets):
        return p_crit(p_outs, p_targets), w_crit(w_outs, w_targets)

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
        dataset = BlockDataset.from_disk(args.dataset_path)
    else:
        words, pos = zip(*load_penn3(args.path, swbd=False))
        word_dict = Dict(
            eos_token=u.EOS, bos_token=u.BOS, force_unk=True,
            max_size=100000)
        pos_dict = Dict(
            eos_token=u.EOS, bos_token=u.BOS, force_unk=False)
        word_dict.fit(words), pos_dict.fit(pos)
        dataset = BlockDataset(
            (pos, words), (pos_dict, word_dict), args.batch_size, 20)
        if args.save_dataset and not os.path.isfile(args.dataset_path):
            dataset.to_disk(args.dataset_path)
    train, valid = dataset.splits(test=None)

    pos_dict, word_dict = train.d

    m = DoubleRNNPOSAwareLM(
        (len(pos_dict.vocab), len(word_dict.vocab)),  # vocabs
        (args.pos_emb_dim, args.word_emb_dim),
        (args.pos_hid_dim, args.word_hid_dim),
        num_layers=(args.pos_num_layers, args.word_num_layers), dropout=0.3)

    if args.gpu:
        m.cuda(), train.set_gpu(args.gpu), valid.set_gpu(args.gpu)

    crit = make_pos_word_criterion(gpu=args.gpu)
    optim = Optimizer(m.parameters(), 'Adam', lr=0.001)
    trainer = POSAwareLMTrainer(
        m, {'train': train, 'valid': valid}, crit, optim)
    trainer.add_loggers(StdLogger())
    trainer.add_hook(
        make_generate_hook(pos_dict, word_dict), num_checkpoints=5)
    trainer.train(args.epochs, args.num_checkpoints)
