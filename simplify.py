
import os

import torch

import utils as u
from dataset import Dataset, Dict
from encoder_decoder import EncoderDecoder
from optimizer import Optimizer
from train import train_model


def load_data(path, exts):
    src_data, trg_data = [], []
    path = os.path.expanduser(path)
    with open(path + exts[0]) as src, open(path + exts[1]) as trg:
        for src_line, trg_line in zip(src, trg):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if src_line and trg_line:
                src_data.append(src_line.split())
                trg_data.append(trg_line.split())
    return src_data, trg_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', default=['redrum'], nargs='*')
    parser.add_argument('--dev_split', default=0.1, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--bidi', action='store_true')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--emb_dim', default=4, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--att_dim', default=64, type=int)
    parser.add_argument('--att_type', default='Bahdanau', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--prefix', default='model', type=str)
    parser.add_argument('--checkpoint', default=500, type=int)
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--learning_rate', default=1., type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--seed', default=1003, type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    path = '~/Downloads/wiki-aligned-data/aligned'
    src_data, trg_data = load_data(path, ('.main', '.simple'))
    src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS)
    src_dict.fit(src_data, trg_data)
    dicts = {'src': src_dict, 'trg': src_dict}
    train, dev = Dataset.splits(
        src_data, trg_data, dicts,
        test=None, batchify=True, batch_size=args.batch_size,
        sort_key=lambda pair: len(pair[0]))
    src_dict = train.dataset.dicts['src'].s2i

    assert len(src_dict) == len(train.dataset.dicts['src'].vocab)

    print(' * vocabulary size. %d' % len(src_dict))
    print(' * number of train batches. %d' % len(train))
    print(' * maximum batch size. %d' % args.batch_size)

    print('Building model...')

    model = EncoderDecoder(
        (args.layers, args.layers), args.emb_dim, (args.hid_dim, args.hid_dim),
        args.att_dim, src_dict, att_type=args.att_type, dropout=args.dropout,
        bidi=args.bidi)
    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)

    print(model)

    train_model(
        model, train, dev, optimizer, args.epochs,
        gpu=args.gpu, targets=args.targets)
