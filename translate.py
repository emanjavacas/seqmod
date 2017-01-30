
import torch
from torch.autograd import Variable

import utils as u


def translate_batch(model, src_batch, max_decode_len):
    seq_len, batch = src_batch.size()
    pad, eos = model.src_dict[u.PAD], model.src_dict[u.EOS]

    # ENCODE
    enc_output, enc_hidden = [], None
    pad_mask = src_batch.data.eq(pad)  # (seq_len x batch) binary
    for t, t_batch in enumerate(src_batch.chunk(seq_len)):
        emb_t = model.src_embedding(t_batch)
        out, enc_hidden = model.encoder(emb_t, hidden=enc_hidden)
        pad_idx_t = pad_mask[t].nonzero().squeeze(1)  # nonzero adds 1 dim
        if pad_idx_t.nelement() > 0:
            # zero entries in batch corresponding to padding
            enc_hidden[0].data.index_fill_(1, pad_idx_t, 0)
            enc_hidden[1].data.index_fill_(1, pad_idx_t, 0)
        enc_output.append(out)
    enc_output = torch.cat(enc_output, 0)

    # DECODE
    num_eos, label_output = 0, []
    dec_out, dec_hidden, att_weights = None, None, []
    # first input (1 x batch)
    prev_output = src_batch[0].new().fill_(eos).unsqueeze(0)
    prev_output = Variable()
    for i in range(max_decode_len):
        # TODO: apply mask on attention weights (0-weight padding)
        prev_emb = model.src_embedding(prev_output)
        dec_out, dec_hidden, att_weight = model.decoder(
            prev_emb, enc_output, enc_hidden,
            init_hidden=dec_hidden, init_output=dec_out, return_weights=True)
        logs = model.project(out)  # (batch x vocab_size)
        prev_output = logs.min(0)[1]  # argmax over log-probs
        label_output.append(prev_output), att_weights.append(att_weight)
        if prev_output == eos:
            num_eos += 1
            if num_eos > 0:  # break translation
                break

    return label_output, att_weights


def translate_checkpoint(model, target='redrum'):
    batch = torch.Tensor([model.src_dict[c] for c in target]).unsqueeze(0)
    int2char = sorted(model.src_dict.items(), key=lambda char, idx: char)
    labels, att_weights = translate_batch(model, batch, len(target) * 4)
    return ''.join(int2char[i[0]] for i in labels)
