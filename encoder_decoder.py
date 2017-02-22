
import torch
import torch.nn as nn
from torch.autograd import Variable

from encoder import Encoder
from decoder import Decoder
from beam_search import Beam
import utils as u


class EncoderDecoder(nn.Module):
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
                 bidi=True,
                 add_prev=True,
                 project_init=False):
        super(EncoderDecoder, self).__init__()
        enc_hid_dim, dec_hid_dim = hid_dim
        enc_num_layers, dec_num_layers = num_layers
        self.cell = cell
        self.add_prev = add_prev
        self.src_dict = src_dict
        src_vocab_size = len(src_dict)
        self.bilingual = bool(trg_dict)
        if self.bilingual:
            self.trg_dict = trg_dict
            trg_vocab_size = len(trg_dict)

        # embedding layer(s)
        self.src_embedding = nn.Embedding(
            src_vocab_size, emb_dim, padding_idx=self.src_dict[u.PAD])
        if self.bilingual:
            self.trg_embedding = nn.Embedding(
                trg_vocab_size, emb_dim, padding_idx=self.trg_dict[u.PAD])
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
        output_size = trg_vocab_size if self.bilingual else src_vocab_size
        self.project = nn.Sequential(
            nn.Linear(dec_hid_dim, output_size),
            nn.LogSoftmax())

    def init_params(self, init_range=0.05):
        for p in self.parameters():
            p.data.uniform_(-init_range, init_range)

    def init_batch(self, src):
        """
        Constructs a first prev batch for initializing the decoder.
        """
        batch, bos = src.size(1), self.src_dict[u.BOS]
        return src.data.new(1, batch).fill_(bos)

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
        emb_inp = self.src_embedding(inp)
        enc_outs, enc_hidden = self.encoder(emb_inp)
        dec_outs, dec_out, dec_hidden = [], None, None
        # cache encoder att projection for bahdanau
        if self.decoder.att_type == 'Bahdanau':
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)
        else:
            enc_att = None
        emb_f = self.trg_embedding if self.bilingual else self.src_embedding
        for prev in trg.chunk(trg.size(0)):
            emb_prev = emb_f(prev).squeeze(0)
            dec_out, dec_hidden, att_weight = self.decoder(
                emb_prev, enc_outs, enc_hidden, out=dec_out,
                hidden=dec_hidden, enc_att=enc_att)
            dec_outs.append(dec_out)
        return torch.stack(dec_outs)

    def translate(self, src, max_decode_len=2):
        pad, eos, bos = \
            self.src_dict[u.PAD], self.src_dict[u.EOS], self.src_dict[u.BOS]
        # encode
        emb = self.src_embedding(src)
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
            prev_emb = self.src_embedding(prev).squeeze(0)
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
        pad, eos, bos = \
            self.src_dict[u.PAD], self.src_dict[u.EOS], self.src_dict[u.BOS]
        gpu = src.is_cuda
        # encode
        emb = self.src_embedding(src)
        enc_outs, enc_hidden = self.encoder(
            emb, compute_mask=False, mask_symbol=pad)
        # decode
        enc_outs = enc_outs.repeat(1, beam_width, 1)
        enc_hidden = (enc_hidden[0].repeat(1, beam_width, 1),
                      enc_hidden[1].repeat(1, beam_width, 1))
        beam = Beam(beam_width, bos, eos, pad, gpu=gpu)
        dec_out, dec_hidden = None, None
        if self.decoder.att_type == 'Bahdanau':
            enc_att = self.decoder.attn.project_enc_outs(enc_outs)
        else:
            enc_att = None
        while beam.active and len(beam) < len(src) * max_decode_len:
            # add seq_len singleton dim (1 x width)
            prev = Variable(beam.get_current_state().unsqueeze(0), volatile=True)
            prev_emb = self.src_embedding(prev).squeeze(0)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, enc_outs, enc_hidden,
                hidden=dec_hidden, out=dec_out)
            # (width x vocab_size)
            logs = self.project(dec_out)
            beam.advance(logs.data)
            # TODO: this doesn't seem to affect the output :-s
            dec_out = u.swap(dec_out, 0, beam.get_source_beam())
            dec_hidden = (u.swap(dec_hidden[0], 1, beam.get_source_beam()),
                          u.swap(dec_hidden[1], 1, beam.get_source_beam()))
        # decode beams
        scores, hyps = beam.decode(n=beam_width)
        return hyps, scores
