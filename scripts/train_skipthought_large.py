
import torch.optim as optim

from seqmod.modules.encoder_decoder import make_rnn_encoder_decoder
from seqmod.misc import PairedDataset, Trainer, Dict, EarlyStopping, Checkpoint
from seqmod.misc import text_processor
from seqmod.misc.loggers import StdLogger
from seqmod import utils as u

from train_skipthought import make_validation_hook, make_report_hook, make_lr_hook

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--path', nargs='+', required=True)
    parser.add_argument('--dict_path', required=True)
    parser.add_argument('--max_size', type=int, default=100000)
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
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_schedule_epochs', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--patience', default=0, type=int)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=1000)
    parser.add_argument('--num_checkpoints', type=int, default=100)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    d = u.load_model(args.dict_path)

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
    # reporting
    logger = StdLogger()
    report_hook = make_report_hook()
    # validation hook
    checkpoint = None
    if not args.test:
        checkpoint = Checkpoint('Skipthought', buffer_size=3).setup(args)
    validation_hook = make_validation_hook(args.patience, checkpoint)
    # lr_hook
    lr_hook = None
    if args.lr_schedule_factor < 1.0:
        lr_hook = make_lr_hook(
            optimizer, args.lr_schedule_factor, args.lr_schedule_checkpoints)

    valid = None

    for epoch in range(args.epochs):
        for idx, path in enumerate(args.path):
            # prepare data subset
            print("Training on subset [{}/{}]: {}".format(idx+1, len(args.path), path))
            print("Loading data...")
            train = u.load_model(path)
            train = PairedDataset(
                train['p1'], train['p2'], {'src': d, 'trg': d},
                batch_size=args.batch_size, fitted=True, gpu=args.gpu)
            if valid is None:
                train, valid = train.splits(dev=None, test=args.dev_split)
            train.sort_()
            # setup trainer
            trainer = Trainer(m, {'train': train, 'valid': valid}, optimizer,
                              losses=('ppl',), max_norm=args.max_norm)
            trainer.add_loggers(logger)
            trainer.add_hook(validation_hook, num_checkpoints=args.num_checkpoints)
            trainer.add_hook(report_hook, num_checkpoints=args.num_checkpoints)
            if args.lr_schedule_factor < 1.0:
                trainer.add_hook(
                    lr_hook, num_checkpoints=args.lr_schedule_num_checkpoints)
            # train
            trainer.train(1, args.checkpoint)

    if not args.test:
        if not u.prompt("Do you want to keep intermediate results? (yes/no)"):
            checkpoint.remove()
