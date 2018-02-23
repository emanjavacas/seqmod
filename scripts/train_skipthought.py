
import random
import copy

import torch.optim as optim

from seqmod.modules.encoder_decoder import make_rnn_encoder_decoder
from seqmod.misc import PairedDataset, Trainer, Dict, EarlyStopping, Checkpoint
from seqmod.misc import text_processor
from seqmod.misc.loggers import StdLogger
from seqmod import utils as u


def load_sents(*paths, root=None, max_len=-1, processor=text_processor()):
    for path in paths:
        if root is not None:
            path = os.path.join(root, path)
        with open(path, 'r+') as f:
            for line in f:
                line = processor(line.strip())
                if max_len > 0 and len(line) >= max_len:
                    while len(line) >= max_len:
                        yield line[:max_len]
                        line = line[max_len:]
                else:
                    yield line


class SkipthoughtDataset(PairedDataset):
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError("{} >= {}".format(idx, self.num_batches))

        b_from, b_to = idx * self.batch_size, (idx+1) * self.batch_size
        src = self._pack(self.data['src'][b_from: b_to], self.d['src'])
        trg = self._pack(self.data['trg'][b_from+1: b_to+1], self.d['trg'])
        return src, trg


def make_validation_hook(patience, checkpoint):
    early_stopping = None
    if patience:
        early_stopping = EarlyStopping(patience)

    def hook(trainer, epoch, batch, check):
        if early_stopping is not None:
            loss = trainer.validate_model()
            trainer.log("validation_end", {"epoch": epoch, "loss": loss.pack()})
            early_stopping.add_checkpoint(loss.reduce(), copy.deepcopy(trainer.model))
            if checkpoint is not None:
                checkpoint.save(model, loss.reduce())

    return hook


def make_report_hook(level='word', items=10):

    def hook(trainer, epoch, batch, checkpoint):
        trainer.log("info", "Generating...")
        # prepare
        dataset, d = trainer.datasets['valid'], trainer.model.decoder.embeddings.d
        batch_size = dataset.batch_size
        dataset.set_batch_size(items)
        # grab random batch
        (src, src_lengths), (trg, _) = dataset[random.randint(0, len(dataset)-1)]
        scores, hyps, _ = trainer.model.translate_beam(src, src_lengths)
        # report
        trues, report = trg.data.transpose(0, 1).tolist(), ''
        for num, (score, hyp, trg) in enumerate(zip(scores, hyps, trues)):
            report += u.format_hyp(score, hyp, num+1, d, level=level, trg=trg)
        trainer.log("info", '\n***' + report + '\n***')
        # reset batch size
        dataset.set_batch_size(batch_size)

    return hook


def make_lr_hook(optimizer, factor, patience, threshold=0.05, verbose=True):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, threshold=threshold,
        verbose=verbose)

    def hook(trainer, epoch, batch_num, checkpoint):
        loss = trainer.validate_model()
        if verbose:
            trainer.log("validation_end", {"epoch": epoch, "loss": loss.pack()})
        scheduler.step(loss.reduce())

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', nargs='+', required=True)
    parser.add_argument('--max_size', type=int, default=20000)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='word')
    parser.add_argument('--dev_split', type=float, default=0.05)
    # model
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--hid_dim', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--encoder_summary', default='mean-max')
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path',
                        default='/home/corpora/word_embeddings/' +
                        'glove.6B.300d.txt')
    parser.add_argument('--reverse', action='store_true')
    # training
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_schedule_epochs', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--patience', default=0, type=int)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=100)
    parser.add_argument('--hooks_per_epoch', type=int, default=2)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    processor = text_processor(
        lower=args.lower, num=args.num, level=args.level)
    train = list(load_sents(*args.path, max_len=args.max_len, processor=processor))
    d = Dict(eos_token=u.EOS, bos_token=u.BOS, unk_token=u.UNK,
             pad_token=u.PAD, max_size=args.max_size, force_unk=True
    ).fit(train)
    train, valid = PairedDataset(
        train, None, {'src': d}, batch_size=args.batch_size, gpu=args.gpu
    ).splits(dev=None, test=args.dev_split)

    print("Building model...")
    m = make_rnn_encoder_decoder(
        args.num_layers, args.emb_dim, args.hid_dim, d, cell=args.cell,
        encoder_summary=args.encoder_summary, dropout=args.dropout,
        reuse_hidden=False, add_init_jitter=True, input_feed=False, att_type=None,
        tie_weights=True, word_dropout=args.word_dropout, reverse=args.reverse)

    print(m)
    print('* number of params: ', sum(p.nelement() for p in m.parameters()))

    u.initialize_model(m)

    if args.init_embeddings:
        m.encoder.embeddings.init_embeddings_from_file(
            args.embeddings_path, verbose=True)

    if args.gpu:
        m.cuda()

    optimizer = getattr(optim, args.optimizer)(m.parameters(), lr=args.lr)
    trainer = Trainer(
        m, {'train': train, 'valid': valid}, optimizer, losses=('ppl',),
        max_norm=args.max_norm)

    checkpoint = None
    if not args.test:
        checkpoint = Checkpoint('Skipthought', buffer_size=3).setup(args)

    trainer.add_loggers(StdLogger())
    trainer.add_hook(make_validation_hook(args.patience, checkpoint),
                     hooks_per_epoch=args.hooks_per_epoch)
    trainer.add_hook(make_report_hook(), hooks_per_epoch=args.hooks_per_epoch)
    if args.lr_schedule_factor < 1.0:
        hook = make_lr_hook(
            optimizer, args.lr_schedule_factor, args.lr_schedule_checkpoints)
        trainer.add_hook(hook, hooks_per_epoch=args.lr_schedule_epochs)

    (best_model, valid_loss), test_loss = trainer.train(
        args.epochs, args.checkpoint)

    if not args.test:
        u.save_checkpoint('./models/', best_model, vars(args), d=d, ppl=test_loss)
        if not u.prompt("Do you want to keep intermediate results? (yes/no)"):
            checkpoint.remove()
