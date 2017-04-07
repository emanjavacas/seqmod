
import re
import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.custom import word_dropout
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules import utils as u

from misc.beam_search import Beam


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
    src_dict: Dict,
        A fitted Dict used to encode the data into integers.
    trg_dict: Dict,
        Same as src_dict in case of bilingual training.
    cell: string,
        Cell type to use. One of (LSTM, GRU).
    att_type: string,
        Attention mechanism to use. One of (Global, Bahdanau).
    dropout: float
    word_dropout: float
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
                 word_dropout=0.0,
                 maxout=0,
                 bidi=True,
                 add_prev=True,
                 tie_weights=False,
                 project_on_tied_weights=False,
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
        self.trg_dict = trg_dict or src_dict
        src_vocab_size = len(self.src_dict)
        trg_vocab_size = len(self.trg_dict)
        self.bilingual = bool(trg_dict)

        # word_dropout
        self.word_dropout = word_dropout
        self.target_code = self.src_dict.get_unk()
        self.reserved = (self.src_dict.get_eos(),
                         self.src_dict.get_bos(),
                         self.src_dict.get_pad())

        # embedding layer(s)
        self.src_embeddings = nn.Embedding(
            src_vocab_size, emb_dim, padding_idx=self.src_dict.get_pad())
        if self.bilingual:
            self.trg_embeddings = nn.Embedding(
                trg_vocab_size, emb_dim, padding_idx=self.trg_dict.get_pad())
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
            project = nn.Linear(emb_dim, output_size)
            project.weight = self.trg_embeddings.weight
            if not project_on_tied_weights:
                assert emb_dim == dec_hid_dim, \
                    "When tying weights, output projection and " + \
                    "embedding layer should have equal size"
                self.project = nn.Sequential(project, nn.LogSoftmax())
            else:
                project_tied = nn.Linear(dec_hid_dim, emb_dim)
                self.project = nn.Sequential(
                    project_tied, project, nn.LogSoftmax())
        else:
            self.project = nn.Sequential(
                nn.Linear(dec_hid_dim, output_size),
                nn.LogSoftmax())

    # General utility functions
    def is_cuda(self):
        """
        Whether the model is on a gpu. We assume no device sharing.
        """
        return next(self.parameters()).is_cuda

    def parameters(self):
        for p in super(EncoderDecoder, self).parameters():
            if p.requires_grad is True:
                yield p

    def n_params(self):
        return sum([p.nelement() for p in self.parameters()])

    # Initializers
    def init_encoder(self, model, layer_map={'0': '0'}, target_module='rnn'):
        merge_map = {}
        for p in model.state_dict().keys():
            if not p.startswith(target_module):
                continue
            from_layer = ''.join(filter(str.isdigit, p))
            if from_layer not in layer_map:
                continue
            s = p.replace(target_module, 'encoder.rnn') \
                 .replace(from_layer, layer_map[from_layer])
            merge_map[p] = s
        state_dict = u.merge_states(
            self.state_dict(), model.state_dict(), merge_map)
        self.load_state_dict(state_dict)

    def init_decoder(self, model, target_module='rnn', layers=(0,)):
        assert isinstance(model.rnn, type(self.decoder.rnn_step))
        target_rnn = getattr(model, target_module).state_dict().keys()
        source_rnn = self.decoder.rnn_step.state_dict().keys()
        merge_map = {}
        for param in source_rnn:
            try:
                # Decoder has format "LSTMCell_0.weight_ih"
                num, suffix = re.findall(r".*([0-9]+)\.(.*)", param)[0]
                # LM rnn has format "weight_ih_l0"
                target_param = suffix + "_l" + num
                if target_param in target_rnn:
                    merge_map[target_param] = "decoder.rnn_step." + param
            except IndexError:
                continue        # couldn't find target module
        state_dict = u.merge_states(
            self.state_dict(), model.state_dict(), merge_map)
        self.load_state_dict(state_dict)

    def init_embedding(self, model,
                       source_module='src_embeddings',
                       target_module='embeddings'):
        state_dict = u.merge_states(
            model.state_dict(), self.state_dict(),
            {target_module: source_module})
        self.load_state_dict(state_dict)

    def init_batch(self, src):
        """
        Constructs a first prev batch for initializing the decoder.
        """
        batch, bos = src.size(1), self.src_dict.get_bos()
        return src.data.new(1, batch).fill_(bos)

    def freeze_submodule(self, module):
        for p in getattr(self, module).parameters():
            p.requires_grad = False

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
        inp = word_dropout(
            inp, self.target_code, reserved_codes=self.reserved,
            dropout=self.word_dropout, training=self.training)
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
        pad = self.src_dict.get_pad()
        eos = self.src_dict.get_eos()
        bos = self.src_dict.get_bos()
        gpu = src.is_cuda
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
        scores, hyp, atts = [], [], []
        prev = Variable(src.data.new([bos]), volatile=True).unsqueeze(0)
        if gpu: prev = prev.cuda()
        for _ in range(len(src) * max_decode_len):
            prev_emb = self.trg_embeddings(prev).squeeze(0)
            dec_out, dec_hidden, att_weights = self.decoder(
                prev_emb, enc_outs, enc_hidden, enc_att=enc_att,
                hidden=dec_hidden, out=dec_out)
            # (batch x vocab_size)
            outs = self.project(dec_out)
            # (1 x batch) argmax over log-probs (take idx across batch dim)
            best_score, prev = outs.max(1)
            prev = prev.t()
            # concat of step vectors along seq_len dim
            scores.append(best_score.squeeze().data[0])
            atts.append(att_weights.squeeze().data.cpu().numpy().tolist())
            hyp.append(prev.squeeze().data[0])
            # termination criterion: decoding <eos>
            if prev.data.eq(eos).nonzero().nelement() > 0:
                break
        # add singleton dimension for compatibility with other decoding
        return [scores], [hyp], [atts]

    def translate_beam(self, src, max_decode_len=2, beam_width=5):
        """
        Translate a single input sequence using beam search.

        Parameters:
        -----------

        src: torch.LongTensor (seq_len x 1)
        """
        pad = self.src_dict.get_pad()
        eos = self.src_dict.get_eos()
        bos = self.src_dict.get_bos()
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
            outs = self.project(dec_out)
            beam.advance(outs.data)
            # TODO: this doesn't seem to affect the output :-s
            dec_out = u.swap(dec_out, 0, beam.get_source_beam())
            if self.cell.startswith('LSTM'):
                dec_hidden = (u.swap(dec_hidden[0], 1, beam.get_source_beam()),
                              u.swap(dec_hidden[1], 1, beam.get_source_beam()))
            else:
                dec_hidden = u.swap(dec_hidden, 1, beam.get_source_beam())
        # decode beams
        scores, hyps = beam.decode(n=beam_width)
        return scores, hyps, None  # TODO: return attention


class ForkableMultiTarget(EncoderDecoder):
    def fork_target(self, **init_opts):
        import copy
        model = copy.deepcopy(self)
        model.freeze_submodule('src_embeddings')
        model.freeze_submodule('encoder')
        model.decoder.apply(u.make_initializer(**init_opts))
        model.project.apply(u.make_initializer(**init_opts))
        return model
