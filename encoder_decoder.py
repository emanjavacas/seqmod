
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from beam_search import Beam
# from beam import Beam
import utils as u


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers, cell,
                 dropout=0.0, bidi=True):
        self.num_layers = num_layers
        self.dirs = 2 if bidi else 1
        self.bidi = bidi
        self.hid_dim = hid_dim // self.dirs
        assert self.hid_dim % self.dirs == 0, \
            "Hidden dimension must be even for BiRNNs"

        super(Encoder, self).__init__()
        self.rnn = getattr(nn, cell)(in_dim, self.hid_dim,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     bidirectional=bidi)

    def init_hidden_for(self, inp):
        size = (self.dirs * self.num_layers, inp.size(1), self.hid_dim)
        h_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        c_0 = Variable(inp.data.new(*size).zero_(), requires_grad=False)
        return h_0, c_0

    def forward(self, inp, hidden=None):
        """
        Paremeters:
        -----------
        inp: torch.Tensor (seq_len x batch x emb_dim)

        hidden: tuple (h_0, c_0)
            h_0: ((num_layers * num_dirs) x batch x hid_dim)
            n_0: ((num_layers * num_dirs) x batch x hid_dim)

        Returns: output, (h_t, c_t)
        --------
        output : (seq_len x batch x hidden_size * num_directions)
            tensor with output features (h_t) in last layer, for each t
            hidden_size = hidden_size * 2 if bidirectional

        h_t : (num_layers * num_directions x batch x hidden_size)
            tensor with hidden state for t=seq_len

        c_t : (num_layers * num_directions x batch x hidden_size)
            tensor containing the cell state for t=seq_len
        """
        enc_outs, hidden = self.rnn(inp, hidden or self.init_hidden_for(inp))
        return enc_outs, hidden


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, dec_out, enc_outs, mask=None, *args, **kwargs):
        """
        dec_out: (batch x hid_dim)
        enc_outs: (seq_len x batch x hid_dim (== att_dim))
        """
        # (batch x att_dim x 1)
        dec_att = self.linear_in(dec_out).unsqueeze(2)
        # (batch x seq_len x att_dim) * (batch x att_dim x 1) -> (batch x seq_len)
        weights = torch.bmm(enc_outs.t(), dec_att).squeeze(2)
        weights = self.softmax(weights)
        if mask is not None:
            weights.data.masked_fill_(mask, -math.inf)
        # (batch x 1 x seq_len) * (batch x seq_len x att_dim) -> (batch x att_dim)
        weighted = weights.unsqueeze(1).bmm(enc_outs.t()).squeeze(1)
        # (batch x att_dim * 2)
        combined = torch.cat([weighted, dec_out], 1)
        output = nn.functional.tanh(self.linear_out(combined))
        return output, weights


class BahdanauAttention(nn.Module):
    def __init__(self, att_dim, enc_hid_dim, dec_hid_dim):
        super(BahdanauAttention, self).__init__()
        self.att_dim = att_dim
        self.enc2att = nn.Linear(enc_hid_dim, att_dim, bias=False)
        self.dec2att = nn.Linear(dec_hid_dim, att_dim, bias=False)
        self.att_v = nn.Parameter(torch.Tensor(att_dim))
        self.softmax = nn.Softmax()

    def project_enc_outs(self, enc_outs):
        """
        mapping: (seq_len x batch x hid_dim) -> (seq_len x batch x att_dim)

        Parameters:
        -----------
        enc_outs: torch.Tensor (seq_len x batch x hid_dim),
            output of encoder over seq_len input symbols

        Returns:
        --------
        enc_att: torch.Tensor (seq_len x batch x att_dim),
            Projection of encoder output onto attention space
        """
        return torch.cat([self.enc2att(i).unsqueeze(0) for i in enc_outs])

    def forward(self, dec_out, enc_outs, enc_att, mask=None, *args, **kwargs):
        """
        Parameters:
        -----------
        dec_out: torch.Tensor (batch x dec_hid_dim)
            Output of decoder at current step

        enc_outs: torch.Tensor (seq_len x batch x enc_hid_dim)
            Output of encoder over the entire sequence

        enc_att: see self.project_enc_outs(self, enc_outs)

        Returns:
        --------
        context: torch.Tensor (batch x hid_dim), weights (batch x seq_len)
            Batch-first matrix of context vectors (for each input in batch)
        """
        # enc_outputs * weights
        # weights: softmax(E) (seq_len x batch)
        # E: att_v (att_dim) * tanh(dec_att + enc_att) -> (seq_len x batch)
        # tanh(dec_out_att + enc_output_att) -> (seq_len x batch x att_dim)
        seq_len, batch, hid_dim = enc_att.size()
        # project current decoder output onto attention (batch_size x att_dim)
        dec_att = self.dec2att(dec_out)
        # elemwise addition of dec_out over enc_att
        # dec_enc_att: (batch x seq_len x att_dim)
        dec_enc_att = nn.functional.tanh(enc_att + u.tile(dec_att, seq_len))
        # dec_enc_att (seq_len x batch x att_dim) * att_v (att_dim)
        #   -> weights (batch x seq_len)
        weights = self.softmax(u.bmv(dec_enc_att.t(), self.att_v).squeeze(2))
        if mask is not None:
            weights.data.masked_fill_(mask, -math.inf)
        # enc_outs: (seq_len x batch x hid_dim) * weights (batch x seq_len)
        #   -> context: (batch x hid_dim)
        context = weights.unsqueeze(1).bmm(enc_outs.t()).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, hid_dim, num_layers, cell,
                 att_dim, att_type='Global', dropout=0.0,
                 add_prev=False, project_init=False):
        """
        Parameters:
        -----------
        project_init: bool, optional
            Whether to use an extra projection on last encoder hidden state to
            initialize decoder hidden state.
        """
        in_dim = enc_hid_dim if not add_prev else enc_hid_dim + emb_dim
        enc_num_layers, dec_num_layers = num_layers
        self.num_layers = dec_num_layers
        self.hid_dim = hid_dim
        self.add_prev = add_prev
        self.project_init = project_init

        super(Decoder, self).__init__()
        # decoder layers
        self.layers = []
        for layer in range(dec_num_layers):
            rnn = getattr(nn, cell + 'Cell')(in_dim, hid_dim)
            # since layer isn't an attribute of this module, we add it manually
            self.add_module('rnn_%d' % layer, rnn)
            self.layers.append(rnn)
            in_dim = hid_dim

        # dropout
        self.has_dropout = bool(dropout)
        if self.has_dropout:
            self.dropout = nn.Dropout(dropout)
        # attention network
        self.att_type = att_type
        if att_type == 'Bahdanau':
            self.attn = BahdanauAttention(att_dim, in_dim, hid_dim)
        elif att_type == 'Global':
            assert att_dim == in_dim == hid_dim, \
                "For global, Encoder, Decoder & Attention must have same size"
            self.attn = GlobalAttention(hid_dim)
        else:
            raise ValueError("unknown attention network [%s]" % att_type)
        # init state matrix (Bahdanau)
        if self.project_init:
            assert self.att_type != "Global", \
                "GlobalAttention doesn't support project_init yet"
            # normally dec_hid_dim == enc_hid_dim, but if not we project
            self.W_h = nn.Parameter(torch.Tensor(
                enc_hid_dim * enc_num_layers, hid_dim * dec_num_layers))
            self.W_c = nn.Parameter(torch.Tensor(
                enc_hid_dim * enc_num_layers, hid_dim * dec_num_layers))

    def rnn_step(self, inp, hidden):
        """
        Parameters:
        -----------

        inp: torch.Tensor (batch x inp_dim),
            Tensor holding the target for the previous decoding step,
            inp_dim = emb_dim or emb_dim + hid_dim if self.add_pred is True.

        hidden: tuple (h_c, c_0), output of previous step or init hidden at 0,
            h_c: (num_layers x batch x hid_dim)
            n_c: (num_layers x batch x hid_dim)

        Returns: output, (h_n, c_n)
        --------
        output: torch.Tensor (batch x hid_dim)
        h_n: torch.Tensor (num_layers x batch x hid_dim)
        c_n: torch.Tensor (num_layers x batch x hid_dim)
        """
        h_0, c_0 = hidden
        h_1, c_1 = [], []  # n refers to layer
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(inp, (h_0[i], c_0[i]))
            h_1.append(h_1_i), c_1.append(c_1_i)
            inp = h_1_i
            # only add dropout to hidden layer (not output)
            if i < (len(self.layers) - 1) and self.has_dropout:
                inp = self.dropout(inp)
        return inp, (torch.stack(h_1), torch.stack(c_1))

    def init_hidden_for(self, enc_hidden):
        """
        Creates a variable at decoding step 0 to be fed as init hidden step

        Returns (h_0, c_0):
        --------
        h_0: torch.Tensor (dec_num_layers x batch x dec_hid_dim)
        c_0: torch.Tensor (dec_num_layers x batch x dec_hid_dim)
        """
        h_t, c_t = enc_hidden
        num_layers, bs, hid_dim = h_t.size()
        size = self.num_layers, bs, self.hid_dim
        # use a projection of last encoder hidden state
        if self.project_init:
            # -> (batch x 1 x num_layers * enc_hid_dim)
            h_t = h_t.t().contiguous().view(-1, num_layers * hid_dim).unsqueeze(1)
            c_t = c_t.t().contiguous().view(-1, num_layers * hid_dim).unsqueeze(1)
            dec_h0 = torch.bmm(h_t, u.tile(self.W_h, bs)).view(*size)
            dec_c0 = torch.bmm(c_t, u.tile(self.W_c, bs)).view(*size)
        else:
            assert num_layers == self.num_layers, \
                "encoder and decoder need equal depth if project_init not set"
            assert hid_dim == self.hid_dim, \
                "encoder and decoder need equal size if project_init not set"
            dec_h0, dec_c0 = enc_hidden
        return dec_h0, dec_c0

    def init_output_for(self, hidden):
        """
        Creates a variable to be concatenated with previous target embedding
        as input for the current rnn step

        Parameters:
        -----------
        hidden: tuple (h_0, c_0)
        h_0: torch.Tensor (dec_num_layers x batch x dec_hid_dim)
        c_0: torch.Tensor (dec_num_layers x batch x dec_hid_dim)

        Returns:
        --------
        torch.Tensor (batch x dec_hid_dim)
        """
        h_0, c_0 = hidden
        data = h_0.data.new(h_0.size(1), self.hid_dim).zero_()
        return Variable(data, requires_grad=False)

    def forward(self, targets, enc_outs, enc_hidden, mask=None,
                init_hidden=None, init_output=None, init_enc_att=None):
        """
        Parameters:
        -----------

        targets: torch.Tensor (seq_len x batch x emb_dim),
            Target output sequence for batch.

        enc_outs: torch.Tensor (seq_len x batch x enc_hid_dim),
            Output of the encoder at the last layer for all encoding steps.

        enc_hidden: tuple (h_t, c_t)
            h_t: (num_layers x batch x hid_dim)
            c_t: (num_layers x batch x hid_dim)
            Can be used to use to specify an initial hidden state for the
            decoder (e.g. the hidden state at the last encoding step.)
        """
        outs, out = [], None
        att_weights, enc_att = [], None
        # init hidden at first decoder lstm layer
        hidden = init_hidden or self.init_hidden_for(enc_hidden)
        # cache encoder att projection for bahdanau
        if self.att_type == 'Bahdanau':
            enc_att = init_enc_att or self.attn.project_enc_outs(enc_outs)
        # first target is just <EOS>
        for y_prev in targets.chunk(targets.size(0)):
            # drop first dim of y_prev (1 x batch X emb_dim)
            y_prev = y_prev.squeeze(0)
            if self.add_prev:
                out = out or init_output or self.init_output_for(hidden)
                dec_inp = torch.cat([out, y_prev], 1)
            else:
                dec_inp = out
            out, hidden = self.rnn_step(dec_inp, hidden)
            out, att_weight = self.attn(out, enc_outs, enc_att, mask=mask)
            if self.has_dropout:
                out = self.dropout(out)
            outs.append(out)
            att_weights.append(att_weight)

        return torch.stack(outs), hidden, torch.stack(att_weights)


class EncoderDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 emb_dim,
                 hid_dim,
                 att_dim,
                 src_dict,
                 tgt_dict=None,
                 cell='LSTM',
                 att_type='Bahdanau',
                 dropout=0.0,
                 bidi=True,
                 add_prev=True,
                 project_init=False):
        super(EncoderDecoder, self).__init__()
        enc_hid_dim, dec_hid_dim = hid_dim
        enc_num_layers, dec_num_layers = num_layers
        self.cell = cell
        self.add_prev = add_prev
        self.bilingual = bool(tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        src_vocab_size = len(src_dict)
        tgt_vocab_size = len(tgt_dict) if tgt_dict else None

        # embedding layer(s)
        self.src_embedding = nn.Embedding(
            src_vocab_size, emb_dim, padding_idx=src_dict[u.PAD])
        if tgt_vocab_size:
            self.tgt_embedding = nn.Embedding(
                tgt_vocab_size, emb_dim, padding_idx=tgt_dict[u.PAD])
        # encoder
        self.encoder = Encoder(
            emb_dim, enc_hid_dim, enc_num_layers,
            cell=cell, bidi=bidi, dropout=dropout)
        # decoder
        self.decoder = Decoder(
            emb_dim, enc_hid_dim, dec_hid_dim, num_layers, cell, att_dim,
            dropout=dropout, add_prev=add_prev, project_init=project_init,
            att_type=att_type)
        # output projection
        self.project = nn.Sequential(
            nn.Linear(dec_hid_dim, tgt_vocab_size or src_vocab_size),
            nn.LogSoftmax())

    def init_params(self, init_range=0.05):
        for p in self.parameters():
            p.data.uniform_(-init_range, init_range)

    def forward(self, inp, tgt):
        """
        Parameters:
        -----------
        inp: torch.Tensor (seq_len x batch),
            Train data for a single batch.
        tgt: torch.Tensor (seq_len x batch)
            Desired output for a single batch

        Returns: outs, hidden, att_ws
        --------
        outs: torch.Tensor (batch x vocab_size),
        hidden: (h_t, c_t)
            h_t: torch.Tensor (batch x dec_hid_dim)
            c_t: torch.Tensor (batch x dec_hid_dim)
        att_weights: (batch x seq_len)
        """
        emb_inp = self.src_embedding(inp)
        enc_outs, hidden = self.encoder(emb_inp)
        if self.encoder.bidi:
            # Repackage hidden in case of bidirectional encoder
            # BiRNN encoder outputs (num_layers * 2 x batch x enc_hid_dim)
            # but decoder expects   (num_layers x batch x dec_hid_dim)
            hidden = (u.repackage_bidi(hidden[0]), u.repackage_bidi(hidden[1]))
        if self.bilingual:
            emb_tgt = self.tgt_embedding(tgt)
        else:
            emb_tgt = self.src_embedding(tgt)
        dec_out, hidden, att_weights = self.decoder(emb_tgt, enc_outs, hidden)
        return dec_out, hidden, att_weights

    def translate(self, src, max_decode_len=2, beam_width=5):
        seq_len, batch = src.size()
        pad, eos = self.src_dict[u.PAD], self.src_dict[u.EOS]
        mask = src.data.eq(pad).t()
        # encode
        emb = self.src_embedding(src)
        enc_outs, enc_hidden = self.encoder(emb)
        if self.encoder.bidi:
            enc_hidden = (u.repackage_bidi(enc_hidden[0]),
                          u.repackage_bidi(enc_hidden[1]))
        # decode
        dec_out, dec_hidden = None, None
        trans = torch.LongTensor(batch).fill_(pad)
        att = torch.FloatTensor(batch)
        prev = Variable(src.data.new(1, batch).fill_(eos), volatile=True)
        for i in range(len(src) * max_decode_len):
            prev_emb = self.src_embedding(prev)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, enc_outs, enc_hidden,
                init_hidden=dec_hidden, init_output=dec_out, mask=mask)
            # (seq x batch x hid) -> (batch x hid)
            dec_out = dec_out.squeeze(0)
            # (batch x vocab_size)
            logs = self.project(dec_out)
            # (1 x batch) argmax over log-probs (take idx across batch dim)
            prev = logs.max(1)[1].t()
            # colwise concat of step vectors
            att = torch.cat([att, att_weights.squeeze(0).data], 1)
            trans = torch.cat([trans, prev.squeeze(0).data], 1)
            # termination criterion: at least one EOS per batch element
            eos_per_batch = trans.eq(eos).sum(1)
            if (eos_per_batch >= 1).sum() == batch:
                break
        return trans[:, 1:].numpy().tolist(), att[:, 1:].numpy().tolist()

    def translate_beam(self, src, max_decode_len=2, beam_width=5):
        seq_len, batch = src.size()
        pad, eos = self.src_dict[u.PAD], self.src_dict[u.EOS]
        # encode
        emb = self.src_embedding(src)
        enc_outs, enc_hidden = self.encoder(emb)
        if self.encoder.bidi:
            enc_hidden = (u.repackage_bidi(enc_hidden[0]),
                          u.repackage_bidi(enc_hidden[1]))
        # decode
        dec_out, dec_hidden = None, None
        # TODO: check gpu
        trans = torch.LongTensor(batch).fill_(pad)
        att = torch.FloatTensor(batch)
        # initialize beam-aware variables
        beams = [Beam(beam_width, eos, pad) for _ in range(batch)]
        enc_outs = u.repeat(enc_outs, (1, beam_width, 1))
        enc_hidden = (u.repeat(enc_hidden[0], (1, beam_width, 1)),
                      u.repeat(enc_hidden[1], (1, beam_width, 1)))
        # repeat only requires same number of elements (but not of dims)
        # for simplicity, we keep the beam width in a separate dim
        mask = src.data.eq(pad).t().unsqueeze(0).repeat(beam_width, 1, 1)
        batch_idx = list(range(batch))
        remaining = batch

        for i in range(len(src) * max_decode_len):
            beam_data = [b.get_current_state() for b in beams if b.active]
            # (1 x batch * width)
            prev = Variable(torch.cat(beam_data).unsqueeze(0), volatile=True)
            # (1 x batch * width x emb_dim)
            prev_emb = self.src_embedding(prev)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, enc_outs, enc_hidden, mask=mask,
                init_hidden=dec_hidden, init_output=dec_out)
            # (seq x batch * width x hid) -> (batch * width x hid)
            dec_out = dec_out.squeeze(0)
            # (batch x width x vocab_size)
            logs = self.project(dec_out).view(remaining, beam_width, -1)
            active = []
            for b in range(batch):
                beam = beams[b]
                if not beam.active:
                    continue
                idx = batch_idx[b]
                beam.advance(logs.data[idx])
                if beam.active:
                    active.append(b)
                # update hidden states to match each beam source
                for dec in dec_hidden:
                    # (layers x width * batch x hid_dim)
                    # -> (layers x width x batch x hid_dim)
                    _, _, hid_dim = dec.size()
                    size = -1, beam_width, remaining, hid_dim
                    beam_hid = dec.view(*size)[:, :, idx]
                    source_beam = beam.get_source_beam()
                    target_hid = beam_hid.data.index_select(1, source_beam)
                    beam_hid.data.copy_(target_hid)

            if not active:
                break

            # TODO: check GPU
            active_idx = torch.LongTensor([batch_idx[k] for k in active])
            batch_idx = {b: idx for idx, b in enumerate(active)}

            def purge(t):
                *head, batch, hid_dim = t.size()
                view = t.data.view(-1, remaining, hid_dim)
                size = *head, batch * len(active_idx) // remaining, hid_dim
                return Variable(view.index_select(1, active_idx).view(*size))

            enc_hidden = (purge(enc_hidden[0]), purge(enc_hidden[1]))
            dec_hidden = (purge(dec_hidden[0]), purge(dec_hidden[1]))
            dec_out, enc_outs = purge(dec_out), purge(enc_outs)
            mask = mask.index_select(1, active_idx)
            remaining = len(active)

        # retrieve best hypothesis
        scores, hyps = [], []
        for b in beams:
            score, hyp = b.decode(n=beam_width)
            scores.append(score), hyps.append(hyp)

        return scores, hyps
