
import torch
import torch.nn as nn
from torch.autograd import Variable

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
        return self.rnn(inp, hidden or self.init_hidden_for(inp))


class GlobalAttention(nn.Module):
    def __init__(self, att_dim, enc_hid_dim, dec_hid_dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(enc_hid_dim, att_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(att_dim + dec_hid_dim, att_dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, inp, context):
        """
        inp: batch x dim
        context: batch x source_seq_len x dim
        """
        target = self.linear_in(inp).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -math.inf)
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        contextCombined = torch.cat((weightedContext, inp), 1)

        contextOutput = self.tanh(self.linear_out(contextCombined))

        return contextOutput, attn


class BahdanauAttention(nn.Module):
    def __init__(self, att_dim, enc_hid_dim, dec_hid_dim):
        super(BahdanauAttention, self).__init__()
        self.att_dim = att_dim
        # self.enc2att = nn.Linear(enc_hid_dim, att_dim, bias=False)
        # self.dec2att = nn.Linear(dec_hid_dim, att_dim, bias=False)
        self.enc2att = nn.Parameter(torch.Tensor(enc_hid_dim, att_dim))
        self.dec2att = nn.Parameter(torch.Tensor(dec_hid_dim, att_dim))
        self.att_v = nn.Parameter(torch.Tensor(att_dim))
        self.softmax = nn.Softmax()

    def project_enc_output(self, enc_outputs):
        """
        mapping: (seq_len x batch x hid_dim) -> (seq_len x batch x att_dim)

        Parameters:
        -----------
        enc_outputs: torch.Tensor (seq_len x batch x hid_dim),
            output of encoder over seq_len input symbols

        Returns:
        --------
        enc_att: torch.Tensor (seq_len x batch x att_dim),
            Projection of encoder output onto attention space
        """
        bs = enc_outputs.size(1)
        # replicate enc2att over batch dimension (batch x hid_dim x att_dim)
        enc2att = u.tile(self.enc2att, bs)
        # batch-first multiply & back to batch-second (seq_len x batch att_dim)
        enc_att = enc_outputs.t().bmm(enc2att).t()
        return enc_att

    def forward(self, dec_output, enc_outputs, enc_att):
        """
        Parameters:
        -----------
        dec_output: torch.Tensor (batch x dec_hid_dim)
            Output of decoder at current step

        enc_outputs: torch.Tensor (seq_len x batch x enc_hid_dim)
            Output of encoder over the entire sequence

        enc_att: see self.project_enc_output(self, enc_outputs)

        Returns:
        --------
        context: torch.Tensor (batch x hid_dim), weights (batch x seq_len)
            Batch-first matrix of context vectors (for each input in batch)
        """
        # enc_outputs * weights
        # weights: softmax(E) (seq_len x batch)
        # E: att_v (att_dim) * tanh(dec_att + enc_att) -> (seq_len x batch)
        # tanh(dec_output_att + enc_output_att) -> (seq_len x batch x att_dim)
        seq_len, batch, hid_dim = enc_att.size()
        # project current decoder output onto attention (batch_size x att_dim)
        dec_att = dec_output.mm(self.dec2att)
        # elemwise addition of dec_out over enc_att
        # dec_enc_att: (batch x seq_len x att_dim)
        dec_enc_att = nn.functional.tanh(enc_att + u.tile(dec_att, seq_len))
        # dec_enc_att (seq_len x batch x att_dim) * att_v (att_dim)
        #   -> weights (batch x seq_len)
        weights = self.softmax(u.bmv(dec_enc_att.t(), self.att_v).squeeze(2))
        # enc_outputs: (seq_len x batch x hid_dim) * weights (batch x seq_len)
        #   -> context: (batch x hid_dim)
        weights = weights.unsqueeze(2)
        context = enc_outputs.t().transpose(2, 1).bmm(weights).squeeze(2)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, hid_dim, num_layers, cell,
                 att_dim, dropout=0.0, add_prev=False, project_init=False):
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
        self.attn = BahdanauAttention(att_dim, in_dim, hid_dim)
        # init state matrix (Bahdanau)
        if self.project_init:
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
                "encoder and decoder need equal depth if project_init is False"
            assert hid_dim == self.hid_dim, \
                "encoder and decoder need equal size if project_init is False"
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

    def forward(self, targets, enc_output, enc_hidden, return_weights=False):
        """
        Parameters:
        -----------

        targets: torch.Tensor (seq_len x batch x emb_dim),
            Target output sequence for batch.

        enc_output: torch.Tensor (seq_len x batch x enc_hid_dim),
            Output of the encoder at the last layer for all encoding steps.

        enc_hidden: tuple (h_t, c_t)
            h_t: (num_layers x batch x hid_dim)
            c_t: (num_layers x batch x hid_dim)
            Can be used to use to specify an initial hidden state for the
            decoder (e.g. the hidden state at the last encoding step.)
        """
        outputs, output = [], None
        if return_weights:
            att_weights = []
        # init hidden at first decoder lstm layer
        hidden = self.init_hidden_for(enc_hidden)
        enc_att = self.attn.project_enc_output(enc_output)
        # first target is just <EOS>
        for y_prev in targets.chunk(targets.size(0)):
            # drop first dim of y_prev (1 x batch X emb_dim)
            y_prev = y_prev.squeeze(0)
            if self.add_prev:
                output = output or self.init_output_for(hidden)
                dec_inp = torch.cat([y_prev, output], 1)
            else:
                dec_inp = y_prev
            output, hidden = self.rnn_step(dec_inp, hidden)
            output, att_weight = self.attn(output, enc_output, enc_att)
            if self.has_dropout:
                output = self.dropout(output)
            outputs.append(output)
            if return_weights:
                att_weights.append(att_weight)

        if return_weights:
            return torch.stack(outputs), hidden, torch.stack(att_weights)
        else:
            return torch.stack(outputs), hidden


class EncoderDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 emb_dim,
                 hid_dim,
                 att_dim,
                 char2int,
                 int2char,
                 cell='LSTM',
                 pad=u.PAD,
                 bidi=True,
                 add_prev=True,
                 dropout=0.0,
                 project_init=False):
        super(EncoderDecoder, self).__init__()
        enc_hid_dim, dec_hid_dim = hid_dim
        enc_num_layers, dec_num_layers = num_layers
        self.cell = cell
        self.add_prev = add_prev
        self.char2int = char2int
        self.int2char = int2char
        vocab_size = len(char2int)

        # embedding layer
        self.embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=char2int[pad])
        # encoder
        self.encoder = Encoder(
            emb_dim, enc_hid_dim, enc_num_layers,
            cell=cell, bidi=bidi, dropout=dropout)
        # decoder
        self.decoder = Decoder(
            emb_dim, enc_hid_dim, dec_hid_dim, num_layers, cell, att_dim,
            dropout=dropout, add_prev=add_prev, project_init=project_init)
        # output projection
        self.out_proj = nn.Sequential(
            nn.Linear(dec_hid_dim, vocab_size),
            nn.LogSoftmax())

        self.init_weights()

    def init_weights(self, init_range=0.05):
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def project(self, dec_t):
        return self.out_proj(dec_t)

    def forward(self, inp, tgt, return_weights=False):
        """
        Parameters:
        -----------
        inp: torch.Tensor (source_seq_len x batch),
            Train data for a single batch.
        tgt: torch.Tensor (target_seq_len x batch)
            Desired output for a single batch

        Returns: outs, hidden, att_ws
        --------
        outs: torch.Tensor (batch x vocab_size),
        hidden: (h_t, c_t)
            h_t: torch.Tensor (batch x dec_hid_dim)
            c_t: torch.Tensor (batch x dec_hid_dim)
        att_ws: (batch x seq_len), batched context vector output by att network
        """
        embedded_inp = self.embedding(inp)
        embedded_tgt = self.embedding(tgt)  # monolingual model
        enc_output, hidden = self.encoder(embedded_inp)
        # repackage_hidden in case of bidirectional encoder
        if self.encoder.bidi:
            h_n, c_n = u.repackage_bidi(hidden[0]), u.repackage_bidi(hidden[1])
            hidden = (h_n, c_n)
        return self.decoder(
            embedded_tgt, enc_output, hidden, return_weights=return_weights)
