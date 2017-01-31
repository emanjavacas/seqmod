
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
        enc_out, enc_hidden = model.encoder(emb_t, hidden=enc_hidden)
        pad_idx_t = pad_mask[t].nonzero()  # nonzero adds 1 dim
        if len(pad_idx_t) != 0 and pad_idx_t.squeeze(1).nelement() > 0:
            # zero entries in batch corresponding to padding
            enc_hidden[0].data.index_fill_(1, pad_idx_t, 0)
            enc_hidden[1].data.index_fill_(1, pad_idx_t, 0)
        enc_output.append(enc_out)
    enc_output = torch.cat(enc_output, 0)
    if model.encoder.bidi:
        enc_hidden = (u.repackage_bidi(enc_hidden[0]),
                      u.repackage_bidi(enc_hidden[1]))

    # DECODE
    eos_batch = torch.LongTensor(batch).fill_(0)
    label_output = []
    dec_out, dec_hidden, att_weights = None, None, []
    # first input (1 x batch)
    prev_output = Variable(
        src_batch[0].data.new(1, batch).fill_(eos), volatile=True)
    for i in range(max_decode_len):
        # TODO: apply mask on attention weights to 0-weight padding
        prev_emb = model.src_embedding(prev_output)
        dec_out, dec_hidden, att_weight = model.decoder(
            prev_emb, enc_output, enc_hidden,
            init_hidden=dec_hidden, init_output=dec_out, return_weights=True)
        dec_out = dec_out.squeeze(0)  # (seq x batch x hid) -> (batch x hid)
        logs = model.project(dec_out)  # (batch x vocab_size)
        prev_output = logs.max(1)[1]  # argmax over log-probs
        label_output.append(prev_output.data), att_weights.append(att_weight)
        eos_batch.set_(prev_output.data.eq(eos).long())
        if eos_batch.sum() >= batch * 2:  # break translation
            break

    return label_output, att_weights


def translate_checkpoint(model, target):
    batch = Variable(
        torch.LongTensor([model.src_dict[c] for c in target]).unsqueeze(1),
        volatile=True)
    int2char = {i: c for c, i in model.src_dict.items()}
    labels, att_weights = translate_batch(model, batch, len(target) * 4)
    return ''.join(int2char[i[0][0]] for i in labels)
