
import copy
import os
import random

from torch import optim

from seqmod.modules.lm import ConditionalLM
from seqmod.misc.dataset import PairedDataset, Dict
from seqmod.misc import EarlyStopping, Trainer, StdLogger, text_processor, Checkpoint
import seqmod.utils as u


def readlines(inputfile, processor):
    with open(inputfile, 'r', newline='\n') as f:
        for line in f:
            line = line.strip()
            sent, *labels = line.split('\t')
            yield labels, processor(sent)


def load_dataset(path, processor, lang_d, batch_size, gpu, cond_dicts=None):
    labels, sents = zip(*list(readlines(path, processor)))
    if not lang_d.fitted:
        lang_d.fit(sents)
        cond_dicts = []
        for label in labels:
            cond_dicts.append(Dict(sequential=False).fit(label))

    num_batches = len(sents) // batch_size
    blabels, bsents = [], []
    for i in range(num_batches):
        blabels.extend(labels[i:num_batches*batch_size:batch_size])
        bsents.extend(sents[i:num_batches*batch_size:batch_size])

    data, d = (bsents,) + tuple(zip(*blabels)), (lang_d,) + tuple(cond_dicts)
    data = PairedDataset(data, None, {'src': d}, batch_size=batch_size, gpu=gpu)

    return data, cond_dicts


def make_clm_hook(sampled_conds, temperature=1, max_seq_len=200, gpu=False,
                  level='token', early_stopping=None, checkpoint=None):
    """
    Make a generator hook for a CLM.
    """

    def hook(trainer, epoch, batch_num, checkpoint):
        trainer.log("info", "Checking training...")

        loss = trainer.validate_model()
        trainer.log("validation_end", {'epoch': epoch, 'loss': loss.pack()})

        if early_stopping is not None:
            trainer.log("info", "Registering early stopping loss...")
            early_stopping.add_checkpoint(
                loss.reduce(), copy.deepcopy(trainer.model).cpu())

        if checkpoint is not None:
            checkpoint.save(trainer.model, loss.reduce())

        d = trainer.model.embeddings.d
        cond_dicts = tuple([emb.d for emb in trainer.model.cond_embs])

        for conds in sampled_conds:
            conds_str = ''
            for idx, (cond_d, sampled_c) in enumerate(zip(cond_dicts, conds)):
                conds_str += (str(cond_d.vocab[sampled_c]) + '; ')
            trainer.log("info", "\n***\nConditions: " + conds_str)
            scores, hyps = trainer.model.generate(
                d, max_seq_len=max_seq_len, gpu=gpu,
                method='sample', temperature=temperature, conds=conds)
            hyps = [u.format_hyp(score, hyp, hyp_num + 1, d, level=level)
                    for hyp_num, (score, hyp) in enumerate(zip(scores, hyps))]
            trainer.log("info", ''.join(hyps) + "\n")
        trainer.log("info", '***\n')

    return hook


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM')
    parser.add_argument('--emb_dim', default=48, type=int)
    parser.add_argument('--cond_dims', default=24)
    parser.add_argument('--hid_dim', default=640, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--mixtures', default=0, type=int)
    parser.add_argument('--sampled_softmax', action='store_true')
    parser.add_argument('--deepout_layers', default=0, type=int)
    parser.add_argument('--deepout_act', default='MaxOut')
    parser.add_argument('--maxouts', default=2, type=int)
    parser.add_argument('--train_init', action='store_true')
    # dataset
    parser.add_argument('--path')
    parser.add_argument('--processed', action='store_true')
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='token')
    parser.add_argument('--dev_split', default=0.1, type=float)
    parser.add_argument('--test_split', default=0.1, type=float)
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--gpu', action='store_true')
    # - optimizer
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--lr_schedule_checkpoints', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--lr_checkpoints_per_epoch', type=int, default=1)
    # - check
    parser.add_argument('--seed', default=None)
    parser.add_argument('--max_seq_len', default=25, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--checkpoint', default=200, type=int)
    parser.add_argument('--hooks_per_epoch', default=5, type=int)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.processed:
        raise NotImplementedError

    else:
        print("Processing datasets...")
        processor = text_processor(
            lower=args.lower, num=args.num, level=args.level)
        d = Dict(max_size=args.max_size, min_freq=args.min_freq,
                 eos_token=u.EOS, force_unk=True)
        cond_dicts = None

        # already split
        if os.path.isfile(os.path.join(args.path, 'train.csv')):
            # train set
            path = os.path.join(args.path, 'train.csv')
            train, cond_dicts = load_dataset(
                path, processor, d, args.batch_size, args.gpu, cond_dicts)
            # test set
            path = os.path.join(args.path, 'test.csv')
            test, _ = load_dataset(
                path, processor, d, args.batch_size, args.gpu, cond_dicts)
            # valid set
            if os.path.isfile(os.path.join(args.path, 'valid.csv')):
                path = os.path.join(args.path, 'valid.csv')
                valid, _ = load_dataset(
                    path, processor, d, args.batch_size, args.gpu, cond_dicts)
            else:
                train, valid = train.splits(dev=None, test=args.dev_split)

        # split, assume input is single file or dir with txt files
        else:
            train, cond_dicts = load_dataset(
                args.path, processor, d, args.batch_size, args.gpu, cond_dicts)
            train, valid, test = train.splits(
                test=args.test_split, dev=args.dev_split)

    print('Building model...')
    m = ConditionalLM(
        args.emb_dim, args.hid_dim, d, cond_dicts=cond_dicts,
        cond_dims=args.cond_dims, num_layers=args.layers, cell=args.cell,
        dropout=args.dropout, tie_weights=args.tie_weights,
        deepout_layers=args.deepout_layers, deepout_act=args.deepout_act,
        word_dropout=args.word_dropout)

    u.initialize_model(m)

    print(m)
    print(' * n parameters. {}'.format(m.n_params()))

    if args.gpu:
        m.cuda()

    if args.optim == 'Adam':
        optimizer = getattr(optim, args.optim)(
            m.parameters(), lr=args.lr, betas=(0., 0.99), eps=1e-5)
    else:
        optimizer = getattr(optim, args.optim)(m.parameters(), lr=args.lr)

    # create trainer
    trainer = Trainer(
        m, {"train": train, "test": test, "valid": valid}, optimizer,
        max_norm=args.max_norm, losses=('bpc' if args.level == 'char' else 'ppl',))

    # hooks
    # - general hook
    early_stopping = None
    if args.patience > 0:
        early_stopping = EarlyStopping(args.patience)

    checkpoint = None
    if args.save:
        checkpoint = Checkpoint(m.__class__.__name__, keep=3).setup(args)

    # (sample conditions if needed)
    sampled_conds = []
    for _ in range(5):
        sample = [d.index(random.sample(d.vocab, 1)[0]) for d in cond_dicts]
        sampled_conds.append(sample)

    model_hook = make_clm_hook(
        sampled_conds, temperature=args.temperature,
        max_seq_len=args.max_seq_len, gpu=args.gpu, level=args.level,
        early_stopping=early_stopping, checkpoint=checkpoint)
    trainer.add_hook(model_hook, hooks_per_epoch=args.hooks_per_epoch)

    # - lr schedule hook
    if args.lr_schedule_factor < 1.0:
        hook = u.make_lr_hook(
            optimizer, args.lr_schedule_factor, args.lr_schedule_checkpoints)
        # run a hook args.checkpoint * 4 batches
        trainer.add_hook(hook, hooks_per_epoch=args.lr_checkpoints_per_epoch)

    # loggers
    trainer.add_loggers(StdLogger())

    (best_model, valid_loss), test_loss = trainer.train(args.epochs, args.checkpoint)
