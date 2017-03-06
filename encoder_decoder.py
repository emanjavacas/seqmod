
import torch
import torch.nn as nn
from torch.autograd import Variable

from modules import TiedEmbedding, TiedLinear
from encoder import Encoder
from decoder import Decoder
from beam_search import Beam
import utils as u


class EncoderDecoder(nn.Module):
    """
    Vanilla configurable encoder-decoder architecture

    Parameters:
    -----------
    num_layers: tuple(enc_num_layers, dec_num_layers) or int,
        Number of layers for both the encoder and the decoder.
    emb_dim: int, embedding dimension
    hid_dim: tuple(enc_hid_dim, dec_hid_dim) or int,
        Hidden state size for the encoder and the decoder
    att_dim: int, hidden state for the attention network.
        Note that it has to be equal to the encoder/decoder hidden
        size when using GlobalAttention.
    src_dict: dict,
        A map from input strings to indices used for indexing the
        training data.
    trg_dict: dict,
        Same as src_dict in case of bilingual training.
    cell: string,
        Cell type to use. One of (LSTM, GRU).
    att_type: string,
        Attention mechanism to use. One of (Global, Bahdanau).
    dropout: float
    bidi: bool,
        Whether to use bidirection encoder or not.
    add_prev: bool,
        Whether to feed back the last decoder prediction as input to
        the decoder for the next step together with the hidden state.
    project_init: bool,
        Whether to use an extra projection on last encoder hidden state to
        initialize decoder hidden state.
    """
    def __init__(self,
                 num_layers,
                 emb_dim,
                 hid_dim,
                 att_dim,
                 src_dict,
                 trg_dict=None,
                 cell='LSTM',
                 att_type='Bahdanau',
                 dropout=0.0,
                 maxout=0,
                 bidi=True,
                 add_prev=True,
                 tie_weights=False,
                 project_init=False):
        super(EncoderDecoder, self).__init__()
        if isinstance(hid_dim, tuple):
            enc_hid_dim, dec_hid_dim = hid_dim
        else:
            enc_hid_dim, dec_hid_dim = hid_dim, hid_dim
        if isinstance(num_layers, tuple):
            enc_num_layers, dec_num_layers = num_layers
        else:
            enc_num_layers, dec_num_layers = num_layers, num_layers
        self.cell = cell
        self.add_prev = add_prev
        self.src_dict = src_dict
        src_vocab_size = len(src_dict)
        self.bilingual = bool(trg_dict)
        if self.bilingual:
            self.trg_dict = trg_dict
            trg_vocab_size = len(trg_dict)

        # embedding layer(s)
        embeddings_weights = None
        if tie_weights and not self.bilingual:
            embeddings_weights = nn.parameter.Parameter(
                torch.randn(src_vocab_size, emb_dim))
            self.src_embeddings = TiedEmbedding(
                src_vocab_size, emb_dim, embeddings_weights,
                padding_idx=self.src_dict[u.PAD])
        else:
            self.src_embeddings = nn.Embedding(
                src_vocab_size, emb_dim, padding_idx=self.src_dict[u.PAD])
        if self.bilingual:
            if tie_weights:
                embeddings_weights = nn.parameter.Parameter(
                    torch.randn(trg_vocab_size, emb_dim))
                self.trg_embeddings == TiedEmbedding(
                    trg_vocab_size, emb_dim, embeddings_weights,
                    padding_idx=self.trg_dict[u.PAD])
            else:
                self.trg_embeddings = nn.Embedding(
                    trg_vocab_size, emb_dim, padding_idx=self.trg_dict[u.PAD])
        else:
            self.trg_embeddings = self.src_embeddings

        # encoder
        self.encoder = Encoder(
            emb_dim, enc_hid_dim, enc_num_layers,
            cell=cell, bidi=bidi, dropout=dropout)

        # decoder
        self.decoder = Decoder(
            emb_dim, enc_hid_dim, dec_hid_dim,
            (enc_num_layers, dec_num_layers), cell, att_dim,
            dropout=dropout, maxout=maxout, add_prev=add_prev,
            project_init=project_init, att_type=att_type)

        # output projection
        output_size = trg_vocab_size if self.bilingual else src_vocab_size
        if tie_weights:
            assert emb_dim == dec_hid_dim, \
                "When tying weights, output projection and " + \
                "embedding layer should have equal size"
            self.project = nn.Sequential(
                TiedLinear(dec_hid_dim, output_size, embeddings_weights),
                nn.LogSoftmax())
        else:
            self.project = nn.Sequential(
                nn.Linear(dec_hid_dim, output_size),
                nn.LogSoftmax())

    def is_cuda(self):
        """
        Whether the model is on a gpu. We assume no device sharing.
        """
        return next(self.parameters()).is_cuda

    def _init_rnn(self, model, layer_map, source_attr, target_attr):
        merge_map = {}
        for p in model.state_dict().keys():
            if not p.startswith(target_attr):
                continue
            from_layer = ''.join(filter(str.isdigit, p))
            if from_layer not in layer_map:
                continue
            s = p.replace(target_attr, source_attr) \
                 .replace(from_layer, layer_map[from_layer])
            merge_map[p] = s
        state_dict = u.merge_states(
            self.state_dict(), model.state_dict(), merge_map)
        self.load_state_dict(state_dict)

    def init_encoder(self, model, layer_map={'0': '0'}, target_attr='rnn'):
        self._init_rnn(model, layer_map, 'encoder.rnn', target_attr)

    def init_decoder(self, model, target_attr='rnn', layers=(0,)):
        # this might be more cumbersome since the decoder rnn is
        # implemented using StackedRNN with layer weights in the form:
        # 'decoder.rnn_step.LSTMCell_0.weight_ih', etc...
        pass

    def init_embedding(self, model, target_attr='embeddings'):
        # todo: support also init for trg_embeddings
        merge_map = {target_attr: 'src_embeddings'}
        state_dict = u.merge_states(
            model.state_dict(), self.state_dict(), merge_map)
        self.load_state_dict(state_dict)

    def init_batch(self, src):
        """
        Constructs a first prev batch for initializing the decoder.
        """
        batch, bos = src.size(1), self.src_dict[u.BOS]
        return src.data.new(1, batch).fill_(bos)

    def freeze_submodule(self, module):
        for p in getattr(self, module).parameters():
            p.requires_grad = False

    def parameters(self):
        for p in super(EncoderDecoder, self).parameters():
            if p.requires_grad is not False:
                yield p

    def forward(self, inp, trg):
        """
        Parameters:
        -----------
        inp: torch.Tensor (seq_len x batch),
            Train data for a single batch.
        trg: torch.Tensor (seq_len x batch)
            Desired output for a single batch

        Returns: outs, hidden, att_ws
        --------
        outs: torch.Tensor (batch x vocab_size),
        hidden: (h_t, c_t)
            h_t: torch.Tensor (batch x dec_hid_dim)
            c_t: torch.Tensor (batch x dec_hid_dim)
        att_weights: (batch x seq_len)
        """
        emb_inp = self.src_embeddings(inp)
        enc_outs, enc_hidden = self.encoder(emb_inp)
        dec_outs, dec_out, dec_hidden = [], None, None
        # cache encoder att projection for bahdanau
        if self.decoder.att_type == 'Bahdanau':
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)
        else:
            enc_att = None
        for prev in trg.chunk(trg.size(0)):
            emb_prev = self.trg_embeddings(prev).squeeze(0)
            dec_out, dec_hidden, att_weight = self.decoder(
                emb_prev, enc_outs, enc_hidden, out=dec_out,
                hidden=dec_hidden, enc_att=enc_att)
            dec_outs.append(dec_out)
        return torch.stack(dec_outs)

    def translate(self, src, max_decode_len=2):
        pad, eos, bos = \
            self.src_dict[u.PAD], self.src_dict[u.EOS], self.src_dict[u.BOS]
        # encode
        emb = self.src_embeddings(src)
        enc_outs, enc_hidden = self.encoder(
            emb, compute_mask=False, mask_symbol=pad)
        # decode
        dec_out, dec_hidden = None, None
        if self.decoder.att_type == 'Bahdanau':
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)
        else:
            enc_att = None
        atts, preds = [], []
        prev = Variable(src.data.new([bos]), volatile=True).unsqueeze(0)
        for _ in range(len(src) * max_decode_len):
            prev_emb = self.trg_embeddings(prev).squeeze(0)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, enc_outs, enc_hidden, enc_att=enc_att,
                hidden=dec_hidden, out=dec_out)
            # (batch x vocab_size)
            logs = self.project(dec_out)
            # (1 x batch) argmax over log-probs (take idx across batch dim)
            prev = logs.max(1)[1].t()
            # concat of step vectors along seq_len dim
            atts.append(att_weights.squeeze().data.cpu().numpy().tolist())
            preds.append(prev.squeeze().data.cpu().numpy().tolist()[0])
            # termination criterion: decoding <eos>
            if prev.data.eq(eos).nonzero().nelement() > 0:
                break
        # add singleton hyp dimension for compatibility with other decoding
        return [preds], atts

    def translate_beam(self, src, max_decode_len=2, beam_width=5):
        """
        Translate a single input sequence using beam search.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x 1)
        """
        pad, eos, bos = \
            self.src_dict[u.PAD], self.src_dict[u.EOS], self.src_dict[u.BOS]
        gpu = src.is_cuda
        # encode
        emb = self.src_embeddings(src)
        enc_outs, enc_hidden = self.encoder(
            emb, compute_mask=False, mask_symbol=pad)
        # decode
        enc_outs = enc_outs.repeat(1, beam_width, 1)
        if self.cell.startswith('LSTM'):
            enc_hidden = (enc_hidden[0].repeat(1, beam_width, 1),
                          enc_hidden[1].repeat(1, beam_width, 1))
        else:
            enc_hidden = enc_hidden.repeat(1, beam_width, 1)
        beam = Beam(beam_width, bos, eos, gpu=gpu)
        dec_out, dec_hidden = None, None
        if self.decoder.att_type == 'Bahdanau':
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)
        else:
            enc_att = None
        while beam.active and len(beam) < len(src) * max_decode_len:
            # add seq_len singleton dim (1 x width)
            prev = Variable(
                beam.get_current_state().unsqueeze(0), volatile=True)
            prev_emb = self.trg_embeddings(prev).squeeze(0)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, enc_outs, enc_hidden, enc_att=enc_att,
                hidden=dec_hidden, out=dec_out)
            # (width x vocab_size)
            logs = self.project(dec_out)
            beam.advance(logs.data)
            # TODO: this doesn't seem to affect the output :-s
            dec_out = u.swap(dec_out, 0, beam.get_source_beam())
            if self.cell.startswith('LSTM'):
                dec_hidden = (u.swap(dec_hidden[0], 1, beam.get_source_beam()),
                              u.swap(dec_hidden[1], 1, beam.get_source_beam()))
            else:
                dec_hidden = u.swap(dec_hidden, 1, beam.get_source_beam())
        # decode beams
        scores, hyps = beam.decode(n=beam_width)
        return hyps, scores
