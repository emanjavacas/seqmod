
from torch.autograd import Variable
import torch

import utils as u


def translate_batch(model, src_batch, max_decode_len):
    seq_len, batch = src_batch.size()
    pad, eos = model.src_dict[u.PAD], model.src_dict[u.EOS]
    # ENCODE (padding)
    # enc_output, enc_hidden = [], None
    # pad_mask = src_batch.data.eq(pad)  # (seq_len x batch) binary
    # for t, t_batch in enumerate(src_batch.chunk(seq_len)):
    #     emb_t = model.src_embedding(t_batch)
    #     enc_out, enc_hidden = model.encoder(emb_t, hidden=enc_hidden)
    #     pad_idx_t = pad_mask[t].nonzero()  # nonzero adds 1 dim
    #     if len(pad_idx_t) != 0 and pad_idx_t.squeeze(1).nelement() > 0:
    #         print("PADDING!")
    #         # zero entries in batch corresponding to padding
    #         enc_hidden[0].data.index_fill_(1, pad_idx_t, 0)
    #         enc_hidden[1].data.index_fill_(1, pad_idx_t, 0)
    #     enc_output.append(enc_out)
    #     if model.encoder.bidi:
    #         enc_hidden = (u.unpackage_bidi(enc_hidden[0]),
    #                       u.unpackage_bidi(enc_hidden[1]))
    # enc_output = torch.cat(enc_output, 0)
    # if model.encoder.bidi:
    #     enc_hidden = (u.repackage_bidi(enc_hidden[0]),
    #                   u.repackage_bidi(enc_hidden[1]))
    
    # ENCODE (normal)
    emb = model.src_embedding(src_batch)
    enc_output, enc_hidden = model.encoder(emb)

    # DECODE
    translation = []
    eos_batch = src_batch.data.new(batch).fill_(0)
    dec_out, dec_hidden, att_weights = None, None, []
    prev_output = Variable(
        src_batch[0].data.new(1, batch).fill_(eos), volatile=True)
    for i in range(max_decode_len):
        # TODO: apply mask on attention weights to 0-weight padding
        prev_emb = model.src_embedding(prev_output)
        dec_out, dec_hidden, att_weight = model.decoder(
            prev_emb, enc_output, enc_hidden,
            init_hidden=dec_hidden, init_output=dec_out)
        dec_out = dec_out.squeeze(0)  # (seq x batch x hid) -> (batch x hid)
        logs = model.project(dec_out)  # (batch x vocab_size)
        prev_output = logs.max(1)[1]   # (1 x batch) argmax over log-probs
        translation.append(prev_output.squeeze(0).data[0])
        att_weights.append(att_weight.squeeze(0))
        eos_batch += prev_output.data.eq(eos).long()
        if eos_batch.sum() >= batch * 2:
            break

    return translation, torch.stack(att_weights)


def translate_checkpoint(model, epoch, b, target, gpu=False):
    batch_data = torch.LongTensor([model.src_dict[c] for c in target])
    batch = Variable(batch_data.unsqueeze(1), volatile=True)
    batch = batch.cuda() if gpu else batch
    labels, att_weights = translate_batch(model, batch, len(target) * 4)
    int2char = {i: c for c, i in model.src_dict.items()}
    pred = ''.join(int2char[i] for i in labels)
    return pred, att_weights


# fig = hinton(self.get_attention_matrix(),
#              xlabels=list(target),
#              ylabels=list(pred.replace(u.EOS, '')))
# plt.savefig('./imgs/%s-%d.png' % (prefix, idx))
# plt.close(fig)
