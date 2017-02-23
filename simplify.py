
import os
import random

seed = 1001
random.seed(seed)

import torch

import utils as u
from dataset import Dataset, Dict
from encoder_decoder import EncoderDecoder
from optimizer import Optimizer
from train import train_model


def sent_processor(language='en'):
    try:
        from normalizr import Normalizr
    except ImportError:
        print("Try installing normalizr")
        return lambda sent: sent

    normalizations = [
        ('replace_emails', {'replacement': '<email>'}),
        ('replace_emojis', {'replacement': '<emoji>'}),
        ('replace_urls', {'replacement': ''})]
    normalizr = Normalizr(language=language)

    import re
    NUM = re.compile('[0-9]+')

    def processor(sent):
        sent = normalizr.normalize(sent, normalizations)
        sent = NUM.sub('<num>', sent)
        return sent

    return processor


def load_data(path, exts, sent_processor=sent_processor()):
    src_data, trg_data = [], []
    path = os.path.expanduser(path)
    with open(path + exts[0]) as src, open(path + exts[1]) as trg:
        for src_line, trg_line in zip(src, trg):
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if sent_processor is not None:
                src_line = sent_processor(src_line)
                trg_line = sent_processor(trg_line)
            if src_line and trg_line:
                src_data.append(src_line.split())
                trg_data.append(trg_line.split())
    return src_data, trg_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--dev_split', default=0.1, type=float)
    parser.add_argument('--max_size', default=10000, type=int)
    parser.add_argument('--min_freq', default=5, type=int)
    parser.add_argument('--bidi', action='store_false')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=264, type=int)
    parser.add_argument('--hid_dim', default=128, type=int)
    parser.add_argument('--att_dim', default=64, type=int)
    parser.add_argument('--att_type', default='Bahdanau', type=str)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--project_init', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--optim', default='RMSprop', type=str)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--learning_rate_decay', default=0.5, type=float)
    parser.add_argument('--start_decay_at', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=5., type=float)
    parser.add_argument('--beam', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    if args.gpu:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    src_data, trg_data = load_data(args.path, ('.main', '.simple'))
    src_dict = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
                    max_size=args.max_size, min_freq=args.min_freq)
    src_dict.fit(src_data, trg_data)
    dicts = {'src': src_dict, 'trg': src_dict}
    train, dev = Dataset.splits(
        src_data, trg_data, dicts,
        test=None, batchify=True, batch_size=args.batch_size, gpu=args.gpu,
        sort_key=lambda pair: len(pair[0]))
    s2i = train.dataset.dicts['src'].s2i

    print(' * vocabulary size. %d' % len(src_dict))
    print(' * number of train batches. %d' % len(train))
    print(' * maximum batch size. %d' % args.batch_size)

    print('Building model...')

    model = EncoderDecoder(
        (args.layers, args.layers), args.emb_dim, (args.hid_dim, args.hid_dim),
        args.att_dim, s2i, att_type=args.att_type, dropout=args.dropout,
        bidi=args.bidi, cell=args.cell, project_init=args.project_init)
    optimizer = Optimizer(
        model.parameters(), args.optim, args.learning_rate, args.max_grad_norm,
        lr_decay=args.learning_rate_decay, start_decay_at=args.start_decay_at)

    model.apply(u.Initializer.make_initializer())
    # model.apply(u.default_weight_init)

    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    print(model)

    target = 'It was also the first animated Disney movie to create a whole' + \
             ' bunch of new sound effects to replace many of their original' + \
             ' classic sounds , which would be used occasionally in later' + \
             ' Disney movies .'

    train_model(model, train, dev, optimizer, src_dict, args.epochs,
                gpu=args.gpu, target=target.split(), beam=args.beam,
                checkpoint=args.checkpoint)
