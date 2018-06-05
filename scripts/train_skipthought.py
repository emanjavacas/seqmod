
import os
import glob

from torch import optim

from seqmod.modules.embedding import Embedding
from seqmod.modules.skipthought import Skipthought
from seqmod.misc import Checkpoint, Trainer, StdLogger
from seqmod.misc import SkipthoughtIter, text_processor
from seqmod import utils


def make_skipthought_hook(checkpoint, scheduler):

    def hook(trainer, hook, batch, check):
        if checkpoint is not None:
            trainer.model.eval()
            checkpoint.save_nlast(trainer.model)
            trainer.model.train()

        if scheduler is not None:
            scheduler.step()

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--paths', required=True)
    parser.add_argument('--dict_path', required=True)
    parser.add_argument('--min_len', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=35)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='word')
    parser.add_argument('--max_items', type=int, default=0)
    # model
    parser.add_argument('--mode', default='prev+post')
    parser.add_argument('--clone', action='store_true')
    parser.add_argument('--softmax', default='full')
    parser.add_argument('--emb_dim', type=int, default=620)
    parser.add_argument('--hid_dim', type=int, default=2400)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cell', default='GRU')
    parser.add_argument('--summary', default='last')
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path',
                        default='/home/corpora/word_embeddings/' +
                        'glove.840B.300d.txt')
    # training
    parser.add_argument('--model', default='skipthought')
    parser.add_argument('--scramble', type=float, default=0.1)
    parser.add_argument('--dropword', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_schedule_checkpoints', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=int(5e+6))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_freq', type=int, default=50000)
    parser.add_argument('--checkpoint', type=int, default=1000)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    embeddings = Embedding.from_dict(utils.load_model(args.dict_path), args.emb_dim)

    checkpoint = None
    if not args.test:
        checkpoint = Checkpoint('Skipthought-{}'.format(args.mode), keep=5)
        checkpoint.setup(args)

    m = Skipthought(
        embeddings, args.mode, cell=args.cell, hid_dim=args.hid_dim,
        num_layers=args.num_layers, summary=args.summary,
        softmax=args.softmax, dropout=args.dropout)

    print("Initializing parameters ...")
    utils.initialize_model(
        m,
        rnn={'type': 'rnn_orthogonal', 'args': {'forget_bias': True}},
        emb={'type': 'uniform_', 'args': {'a': -0.1, 'b': 0.1}})

    if args.init_embeddings:
        embeddings.init_embeddings_from_file(args.embeddings_path, verbose=True)

    m.to(device=args.device)

    optimizer = getattr(optim, args.optim)(m.parameters(), lr=args.lr)
    losses = [{'loss': loss, 'format': 'ppl'} for loss in ('prev', 'same', 'post')]
    trainer = Trainer(m, {}, optimizer, max_norm=args.max_norm, losses=losses)
    # logger
    outputfile = None
    if checkpoint is not None:
        outputfile = checkpoint.checkpoint_path()
    trainer.add_loggers(StdLogger(outputfile=outputfile))
    # hook
    scheduler = None
    if args.lr_schedule_factor < 1:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, args.lr_schedule_epochs, args.lr_schedule_factor)
    num_checkpoints = args.save_freq // args.checkpoint
    trainer.add_hook(
        make_skipthought_hook(checkpoint, scheduler), num_checkpoints=num_checkpoints)

    # dataset
    paths = glob.glob(os.path.expanduser(args.paths))
    includes = ('prev' in args.mode, 'same' in args.mode, 'post' in args.mode)
    processor = text_processor(max_len=args.max_len, min_len=args.min_len)
    data = SkipthoughtIter(embeddings.d, *paths, always_reverse=args.clone,
                           includes=includes, processor=processor, device=args.device,
                           verbose=True)
    generator = data.batch_generator(args.batch_size, buffer_size=int(5e+6))

    print()
    print("Training model...")
    trainer.train_generator(args.epochs, generator, args.checkpoint)
