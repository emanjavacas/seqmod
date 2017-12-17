

class BaseDecoder(nn.Module):
    """
    Base abstract class
    """
    def forward(self, inp, enc_out, enc_hidden, **kwargs):
        """
        Parameters:
        -----------
        inp: torch.LongTensor(seq_len x batch)
        enc_out: summary vector from the encoder
        enc_hidden: hidden state of encoder (needed when initializing RNN
            decoder with last RNN encoder hidden step)
        """
        raise NotImplementedError
    

class RNNDecoder(nn.Module):
    """
    RNNDecoder

    Parameters:
    -----------

    - input_feed: bool, whether to concatenate last attentional vector
        to current rnn input. (See Luong et al. 2015). Mostly useful
        for attentional models.
    """
    def __init__(self, embeddings, hid_dim, num_layers, cell,
                 dropout=0.0, input_feed=False, att_type=None,
                 deepout_layers=0, deepout_act='ReLU', tie_weights=False,
                 train_init=False, add_init_jitter=False, reuse_hidden=True):
        self.embeddings = embeddings
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.cell = cell
        self.dropout = dropout
        self.input_feed = input_feed
        self.att_type = att_type
        self.train_init = train_init
        self.add_init_jitter = add_init_jitter
        self.reuse_hidden = reuse_hidden
        super(Decoder, self).__init__()

        emb_dim = self.embeddings.embedding_size
        in_dim = emb_dim if not input_feed else hid_dim + emb_dim

        # rnn layer
        self.rnn_step = self._build_rnn(num_layers, in_dim, hid_dim, dropout)

        # train init
        if self.train_init:
            init_size = self.num_layers, 1, self.hid_dim
            self.h_0 = nn.Parameter(torch.Tensor(*init_size).zero_())

        # attention network (optional)
        if self.att_type is not None:
            self.attn = attn.Attention(
                self.hid_dim, self.hid_dim, scorer=self.att_type)
        self.has_attention = hasattr(self, 'attn')

        # output projection
        self.proj = self._build_projection(
            embeddings, hid_dim, deepout_layers, deepout_act)

    def _build_rnn(self, in_dim, hid_dim, cell, dropout):
        stacked = StackedLSTM if cell == 'LSTM' else StackedGRU
        return stacked(num_layers, in_dim, hid_dim, dropout=dropout)

    def init_hidden_for(self, enc_hidden):
        """
        Creates a variable to be fed as init hidden step.

        Returns:
        --------
        torch.Tensor(num_layers x batch x hid_dim)
        """
        # unpack
        if self.cell.startswith('LSTM'):
            h_0, _ = enc_hidden
        else:
            h_0 = enc_hidden

        # compute h_0
        if not self.reuse_hidden:
            h_0 = h_0.zeros_like(h_0)

        if self.train_init:
            h_0 = self.h_0.repeat(1, h_0.size(1), 1)

        if self.add_init_jitter:
            h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

        # pack
        if self.cell.startswith('LSTM'):
            return h_0, h_0.zeros_like(h_0)
        else:
            return h_0

    def init_output_for(self, hidden):
        """
        Creates a variable to be concatenated with previous target
        embedding as input for the first rnn step. This is used
        for the first decoding step when using the input_feed flag.

        Returns:
        --------
        torch.Tensor(batch x hid_dim)
        """
        if self.cell.startswith('LSTM'):
            hidden = hidden[0]

        _, batch, hid_dim = hidden.size()

        output = torch.normal(hidden.data.new(batch, hid_dim).zero_(), 0.3)

        return Variable(output, volatile=not self.training)

    def forward(self, inp, outs, hidden, prev_out=None, **kwargs):
        """
        Parameters:
        -----------

        inp: torch.FloatTensor(batch x emb_dim), Embedding input
        outs: summary vector(s) from the Encoder
        hidden: previous hidden decoder state
        prev_out: (optional), prev decoder output. Needed for input feeding.
        """
        if self.input_feed:
            if prev_out is None:
                prev_out = self.init_output_for(hidden)
            inp = torch.cat([inp, prev_out], 1)

        out, hidden = self.rnn_step(inp, hidden)

        weight = None
        if self.has_attention:
            out, weight = self.attn(out, outs, **kwargs)

        return out, hidden, weight


class ConditionalRNNDecoder(RNNDecoder):
    def __init__(self, *args, cond_dims, cond_vocabs, **kwargs):
        super(ConditionalRNNDecoder, self).__init__(*args, **kwargs)

        cond_dim = 0       # accumulate conditional emb dims
        self.cond_embs = nn.ModuleList()

        for cond_dim, cond_vocab in zip(cond_dims, cond_vocabs):
            self.cond_embs.append(nn.Embedding(cond_vocab, cond_dim))
            cond_dim += cond_dim

        # overwrite rnn with modified input
        in_dim = self.rnn_step.input_size + cond_dim
        self.rnn_step = self._build_rnn(
            self.num_layers, in_dim, self.hid_dim, self.dropout)

    def forward(self, inp, outs, hidden, conds=None, **kwargs):
        if conds is None:
            raise ValueError("ConditionalRNNDecoder requires `conds`")

        conds = [emb(cond) for cond, emb in zip(conds, self.cond_embs)]
        inp = self.embeddings(inp)
        inp = torch.cat([inp, *conds], 1)

        return super(ConditionalRNNDecoder, self).forward(
            self, inp, outs, hidden, **kwargs)

