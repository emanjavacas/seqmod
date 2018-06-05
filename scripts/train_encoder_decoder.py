
import copy
import string

import random; random.seed(1001)

import torch
try:
    torch.manual_seed(1001)
    torch.cuda.manual_seed(1001)
except:
    print('no NVIDIA driver found')

from torch import optim

from seqmod.modules.encoder_decoder import make_rnn_encoder_decoder
import seqmod.utils as u

from seqmod.misc import EarlyStopping, Trainer, Checkpoint
from seqmod.misc import StdLogger, VisdomLogger, TensorboardLogger
from seqmod.misc import PairedDataset, Dict, inflection_sigmoid

import dummy as d


def translate(model, src, src_lengths, beam=True):
    if beam:
        scores, hyps, _ = model.translate_beam(
            src, src_lengths, beam_width=5, max_decode_len=4)
    else:
        scores, hyps, _ = model.translate(src, src_lengths, max_decode_len=4)

    return scores, hyps


def make_encdec_hook(target, beam=True, max_items=10):

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Translating {}".format(target))
        # prepare
        dataset, d = trainer.datasets['valid'], trainer.model.decoder.embeddings.d
        items = min(max_items, dataset.batch_size)
        # sample random batch
        batch = dataset[random.randint(0, len(dataset) - 1)]
        (src, src_lengths), (trg, _) = batch
        src, src_lengths, trg = src[:, :items], src_lengths[:items], trg[:, :items]
        # translate
        scores, hyps = translate(trainer.model, src, src_lengths, beam=beam)
        # report
        trues, report = trg.transpose(0, 1).tolist(), ''
        for num, (score, hyp, trg) in enumerate(zip(scores, hyps, trues)):
            report += u.format_hyp(score, hyp, num+1, d, level='char', trg=trg)

        trainer.log("info", '\n***' + report + '\n***')

    return hook


def make_att_hook(target, beam=False):
    assert not beam, "beam doesn't output attention yet"

    def hook(trainer, epoch, batch_num, checkpoint):
        d = train.decoder.embedding.d
        scores, hyps, atts = translate(trainer.model, target, beam=beam)
        trainer.log("attention",
                    {"att": atts[0],
                     "score": sum(scores[0]) / len(hyps[0]),
                     "target": [d.bos_token] + list(target),
                     "hyp": ' '.join([d.vocab[i] for i in hyps[0]]).split(),
                     "epoch": epoch,
                     "batch_num": batch_num})

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--path', type=str)
    parser.add_argument('--train_len', default=10000, type=int)
    parser.add_argument('--vocab', default=list(string.ascii_letters) + [' '])
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--max_len', default=25, type=int)
    parser.add_argument('--sample_fn', default='reverse', type=str)
    parser.add_argument('--dev', default=0.1, type=float)
    # model
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM', type=str)
    parser.add_argument('--emb_dim', default=24, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--att_type', default='general', type=str)
    parser.add_argument('--encoder_summary', default='full')
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--input_feed', action='store_true')
    parser.add_argument('--tie_weights', action='store_true')
    # training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_norm', default=10., type=float)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--use_schedule', action='store_true')
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--hooks_per_epoch', default=2, type=int)
    parser.add_argument('--target', default='redrum', type=str)
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    vocab = args.vocab
    size = args.train_len
    batch_size = args.batch_size
    sample_fn = getattr(d, args.sample_fn)

    if args.path is not None:
        with open(args.path, 'rb+') as f:
            dataset = PairedDataset.from_disk(f)
        dataset.set_batch_size(args.batch_size)
        dataset.set_device(args.device)
        train, valid = dataset.splits(sort_by='src', dev=args.dev, test=None)
        src_dict = dataset.dicts['src']
    else:
        str_generator = d.generate_set(
            size, vocab, args.min_len, args.max_len, sample_fn)
        src, trg = zip(*str_generator)
        src, trg = list(map(list, src)), list(map(list, trg))
        src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS)
        src_dict.fit(src, trg)
        trg_dict = src_dict
        if args.reverse:
            trg_dict = copy.deepcopy(src_dict)
            trg_dict.align_right = True
        train, valid = PairedDataset(
            src, trg, {'src': src_dict, 'trg': trg_dict},
            batch_size=args.batch_size, device=args.device,
        ).splits(dev=args.dev, test=None, sort=True)

    print(' * vocabulary size. {}'.format(len(src_dict)))
    print(' * number of train batches. {}'.format(len(train)))
    print(' * maximum batch size. {}'.format(batch_size))

    print('Building model...')

    model = make_rnn_encoder_decoder(
        args.layers, args.emb_dim, args.hid_dim, src_dict, cell=args.cell,
        bidi=True, encoder_summary=args.encoder_summary, att_type=args.att_type,
        reuse_hidden=args.att_type.lower() != 'none',
        dropout=args.dropout, input_feed=args.input_feed,
        word_dropout=args.word_dropout, deepout_layers=args.deepout_layers,
        tie_weights=args.tie_weights, reverse=args.reverse)

    # model.freeze_submodule('encoder')
    # model.encoder.register_backward_hook(u.log_grad)
    # model.decoder.register_backward_hook(u.log_grad)

    u.initialize_model(
        model, rnn={'type': 'orthogonal', 'args': {'gain': 1.0}}
    )

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    print(model)
    print()
    print('* number of parameters: {}'.format(model.n_params()))

    model.to(device=args.device)

    early_stopping = EarlyStopping(args.patience)
    trainer = Trainer(
        model, {'train': train, 'valid': valid}, optimizer, losses=('ppl',),
        early_stopping=early_stopping, max_norm=args.max_norm,
        checkpoint=Checkpoint('EncoderDecoder', mode='nlast', keep=3).setup(args))
    trainer.add_loggers(StdLogger())
    # trainer.add_loggers(VisdomLogger(env='encdec'))
    trainer.add_loggers(TensorboardLogger(comment='encdec'))

    hook = make_encdec_hook(args.target, beam=args.beam)
    trainer.add_hook(hook, hooks_per_epoch=args.hooks_per_epoch)
    hook = u.make_schedule_hook(
        inflection_sigmoid(len(train) * 2, 1.75, inverse=True))
    trainer.add_hook(hook, hooks_per_epoch=1000)

    (model, valid_loss), test_loss = trainer.train(
        args.epochs, args.checkpoint, shuffle=True,
        use_schedule=args.use_schedule)
