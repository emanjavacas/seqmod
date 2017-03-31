
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils as u
import attention
from beam_search import Beam


class Attention(nn.Module):
    def __init__(self, att_dim, hid_dim, emb_dim, num_layers=1):
        self.att_dim = att_dim * num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        super(Attention, self).__init__()
        self.hidden_att = nn.Linear(hid_dim, att_dim)    # W
        self.emb_att = nn.Linear(emb_dim, att_dim)       # U
        self.att_v = nn.Parameter(torch.Tensor(att_dim))  # b

    def forward(hidden, emb):
        """
        Parameters:
        -----------
        hidden: torch.Tensor
            (num_layers * num_dirs x batch_size x hid_dim)
            Hidden state at step h_{t-1}
        emb: torch.Tensor (seq_len x batch_size x emb_dim)
            Embeddings of input words up to step t-1

        Returns: context, weights
        --------
        context: torch.Tensor 
        weights: torch.Tensor
        """
        # hidden                  # need to reshape this to account for num_layers
        hidden_att, emb_att = self.hidden_att(hidden), self.emb_att(emb)



class LM(nn.Module):
    """
    Vanilla RNN-based language model.

    Parameters:
    ===========
    - vocab: int, vocabulary size.
    - emb_dim: int, embedding size,
        This value has to be equal to hid_dim if tie_weights is True and
        project_on_tied_weights is False, otherwise input and output
        embedding dimensions wouldn't match and weights cannot be tied.
    - hid_dim: int, hidden dimension of the RNN.
    - num_layers: int, number of layers of the RNN.
    - cell: str, one of GRU, LSTM.
    - bias: bool, whether to include bias in the RNN.
    - dropout: float, amount of dropout to apply in between layers.
    - tie_weights: bool, whether to tie input and output embedding layers.
    - project_on_tied_weights: bool,
        In case of unequal emb_dim and hid_dim values this option has to
        be True if tie_weights is True. A linear project layer will be
        inserted after the RNN to match back to the embedding dimension.
    """
    def __init__(self, vocab, emb_dim, hid_dim, num_layers=1,
                 cell='GRU', bias=True, dropout=0.0,
                 tie_weights=False, project_on_tied_weights=False,
                 add_attn=False, att_type=None, att_dim=None):
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.tie_weights = tie_weights
        self.project_on_tied_weights = project_on_tied_weights
        if tie_weights and not project_on_tied_weights:
            assert self.emb_dim == self.hid_dim, \
                "When tying weights, output projection and " + \
                "embedding layer should have equal size"
        self.num_layers = num_layers
        self.cell = cell
        self.bias = bias
        self.has_dropout = bool(dropout)
        self.dropout = dropout
        super(LM, self).__init__()

        # input embeddings
        self.embeddings = nn.Embedding(vocab, self.emb_dim)
        # rnn
        self.rnn = getattr(nn, cell)(
            self.emb_dim, self.hid_dim,
            num_layers=num_layers, bias=bias, dropout=dropout)
        # attention
        if add_attn:
            assert self.cell == 'RNN', \
                "currently only RNN supports attention"
            assert att_dim is not None, "Need to specify att_dim"
            self.att_dim = att_dim
            self.attn = Attention(self.att_dim, self.hid_dim, self.emb_dim)
        # output embeddings
        proj_dim = self.hid_dim if add_attn else self.hid_dim
        if tie_weights:
            if self.emb_dim == proj_dim:
                self.project = nn.Linear(proj_dim, vocab)
                self.project.weight = self.embeddings.weight
            else:
                assert project_on_tied_weights, \
                    "Unequal tied layer dims but no projection layer"
                project = nn.Linear(self.emb_dim, vocab)
                project.weight = self.embeddings.weight
                self.project = nn.Sequential(
                    nn.Linear(proj_dim, self.emb_dim), project)
        else:
            self.project = nn.Linear(proj_dim, vocab)

    def parameters(self):
        for p in super(LM, self).parameters():
            if p.requires_grad is True:
                yield p

    def n_params(self):
        return sum([p.nelement() for p in self.parameters()])

    def freeze_submodule(self, module, flag=False):
        for p in getattr(self, module).parameters():
            p.requires_grad = flag

    def init_hidden_for(self, inp):
        batch = inp.size(1)
        size = (self.num_layers, batch, self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        if self.cell.startswith('GRU'):
            return h_0
        else:
            c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
            return h_0, c_0

    def generate_beam(
            self, bos, eos, max_seq_len=20, width=5, gpu=False, **kwargs):
        "Generate text using beam search decoding"
        self.eval()
        beam = Beam(width, bos, eos, gpu=gpu)
        hidden = None
        while beam.active and len(beam) < max_seq_len:
            prev = Variable(
                beam.get_current_state().unsqueeze(0), volatile=True)
            outs, hidden = self(prev, hidden=hidden, **kwargs)
            logs = F.log_softmax(outs)
            beam.advance(logs.data)
            if self.cell.startswith('LSTM'):
                hidden = (u.swap(hidden[0], 1, beam.get_source_beam()),
                          u.swap(hidden[1], 1, beam.get_source_beam()))
            else:
                hidden = u.swap(hidden, 1, beam.get_source_beam())
        scores, hyps = beam.decode(n=width)
        return scores, hyps

    def generate(self, bos, eos, max_seq_len=20, gpu=False, **kwargs):
        "Generate text using simple argmax decoding"
        self.eval()
        prev = Variable(torch.LongTensor([bos]).unsqueeze(0), volatile=True)
        if gpu: prev = prev.cuda()
        hidden, hyp, scores = None, [], []
        for _ in range(max_seq_len):
            outs, hidden = self(prev, hidden=hidden, **kwargs)
            outs = F.log_softmax(outs)
            best_score, prev = outs.max(1)
            prev = prev.t()
            hyp.append(prev.squeeze().data[0])
            scores.append(best_score.squeeze().data[0])
            if prev.data.eq(eos).nonzero().nelement() > 0:
                break
        return [scores], [hyp]

    def predict_proba(self, inp, gpu=False, **kwargs):
        self.eval()
        inp_vec = Variable(torch.LongTensor([inp]), volatile=True)
        if gpu:
            inp_vec.cuda()
        outs, hidden = self(inp_vec, **kwargs)
        outs = u.select_cols(F.log_softmax(outs), inp).sum()
        return outs.data[0] / len(inp)

    def forward(self, inp, hidden=None, **kwargs):
        """
        Parameters:
        -----------
        inp: torch.Tensor (seq_len x batch_size)
        """
        emb = self.embeddings(inp)
        if self.has_dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        outs, hidden = self.rnn(emb, hidden or self.init_hidden_for(emb))
        if self.has_dropout:
            outs = F.dropout(outs, p=self.dropout, training=self.training)
        seq_len, batch, hid_dim = outs.size()
        if hasattr(self, 'attn'):
            context, weights = self.attn(hidden, emb)
        outs = self.project(outs.view(seq_len * batch, hid_dim))
        return outs, hidden


class ForkableLM(LM):
    """
    A LM model that allows to create forks of the current instance with
    frozen Embedding (and eventually RNN) layers for fine tunning the
    non-frozen parameters to particular dataset.
    The parent cannot have the projection layer for tied embeddings,
    since tied layers don't fit in this setup.
    """
    def __init__(self, *args, **kwargs):
        super(ForkableLM, self).__init__(*args, **kwargs)

    def fork_model(self, freeze_rnn=True):
        """
        Creates a child fork from the current model with frozen input
        embeddings (and eventually also frozen RNN).

        Parameters:
        ===========
        - freeze_rnn: optional, whether to also freeze the child RNN layer.
        """
        model = ForkableLM(
            self.vocab, self.emb_dim, self.hid_dim, num_layers=self.num_layers,
            cell=self.cell, dropout=self.dropout, bias=self.bias,
            tie_weights=False, project_on_tied_weights=False)
        source_dict, target_dict = self.state_dict(), model.state_dict()
        target_dict['embeddings.weight'] = source_dict()['embeddings.weight']
        for layer, p in source_dict.items():
            if layer.startswith('project') and \
               self.tie_weights and \
               self.project_on_tied_weights:
                print("Warning: Forked model couldn't use projection layer " +
                      "of parent for the initialization of layer [%s]" % layer)
                continue
            else:
                target_dict[layer] = p
        model.load_state_dict(target_dict)
        model.freeze_submodule('embeddings')
        if freeze_rnn:
            model.freeze_submodule('rnn')
        return model


class MultiheadLM(LM):
    """
    A variant LM that has multiple output embeddings (one for each of a
    given number of heads). This allows the model to fine tune different
    output distribution on different datasets.
    """
    def __init__(self, *args, heads=(), **kwargs):
        super(MultiheadLM, self).__init__(*args, **kwargs):
        assert heads, "MultiheadLM requires at least 1 head but got 0"
        import copy
        if hasattr(self, 'attn'):
            attn = self.attn
            self.attn = {}
        project = self.project
        self.project = {}
        for head in heads:
            project_module = copy.deepcopy(project)
            self.add_module(head, project_module)
            self.project[head] = project_module
            if hasattr(self, 'attn'):
                attn_module = copy.deepcopy(attn)
                self.add_module(head, attn_module)
                self.attn[head] = attn_module

    def forward(self, inp, hidden=None, head=None):
        emb = self.embeddings(inp)
        if self.has_dropout:
            emb = F.dropout(emb, p=self.dropout, training=self.training)
        outs, hidden = self.rnn(emb, hidden or self.init_hidden_for(emb))
        if self.has_dropout:
            outs = F.dropout(outs, p=self.dropout, training=self.training)
        seq_len, batch, hid_dim = outs.size()
        # (seq_len x batch x hid) -> (seq_len * batch x hid)
        outs = self.project[head](outs.view(seq_len * batch, hid_dim))
        return outs, hidden

    @classmethod
    def from_pretrained_model(cls, that_model, heads, **kwargs):
        """
        Create a multihead model from a pretrained LM initializing all weights
        to the LM states and all heads to the same output projection layer
        weights of the parent.
        """
        assert isinstance(that_model, LM)
        this_model = cls(
            that_model.vocab, that_model.emb_dim, that_model.hid_dim,
            num_layers=that_model.num_layers, cell=that_model.cell,
            bias=that_model.bias, dropout=that_model.dropout, heads=heads,
            **kwargs)
        this_state_dict = this_model.state_dict()
        for p, w in that_model.state_dict().items():
            if p in this_state_dict:
                this_state_dict[p] = w
            else:               # got project or attn layer
                *_, weight = p.split('.')
                for head in this_model.heads:
                    this_state_dict[head + "." + weight] = w
        this_model.load_state_dict(this_state_dict)
        return this_model
