
import random
from collections import Counter

import torch


def shuffled(data):
    data = list(data)
    random.shuffle(data)
    return data


def shuffle_pairs(pair1, pair2):
    return zip(*shuffled(zip(pair1, pair2)))


def cumsum(seq):
    seq = [0] + list(seq)
    subseqs = (seq[:i] for i in range(1, len(seq)+1))
    return [sum(subseq) for subseq in subseqs]


def batchify(examples, pad_token=None, align_right=False):
    max_length = max(len(x) for x in examples)
    out = torch.LongTensor(len(examples), max_length) \
               .fill_(pad_token or 0)
    for i in range(len(examples)):
        example = torch.Tensor(examples[i])
        example_length = example.size(0)
        offset = max_length - example_length if align_right else 0
        out[i].narrow(0, offset, example_length).copy_(example)
    out = out.t().contiguous()
    return out


class Dict(object):
    def __init__(self, pad_token=None, eos_token=None, bos_token=None,
                 unk_token='<unk>', max_size=None, min_freq=1,
                 sequential=True):
        """
        Dict
        """
        self.counter = Counter()
        self.vocab = [t for t in [pad_token, eos_token, bos_token] if t]
        self.fitted = False
        self.has_unk = False    # only index unk_token if needed
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        self.max_size = max_size
        self.min_freq = min_freq
        self.sequential = sequential

    def __len__(self):
        return len(self.vocab)

    def get_pad(self):
        return self.s2i.get(self.pad_token, None)

    def get_eos(self):
        return self.s2i.get(self.eos_token, None)

    def get_bos(self):
        return self.s2i.get(self.bos_token, None)

    def get_unk(self):
        return self.s2i.get(self.unk_token, None)

    def _get_unk(self):
        if self.unk_token not in self.s2i:
            unk_code = self.s2i[self.unk_token] = len(self.vocab)
            self.vocab += [self.unk_token]
            self.has_unk = True
            return unk_code
        else:
            return self.s2i[self.unk_token]

    def index(self, s):
        assert self.fitted, "Attempt to index without fitted data"
        if s not in self.s2i:
            return self._get_unk()
        else:
            return self.s2i[s]

    def partial_fit(self, *args):
        for dataset in args:
            for example in dataset:
                self.counter.update(example if self.sequential else [example])

    def fit(self, *args):
        self.partial_fit(*args)
        most_common = self.counter.most_common(self.max_size)
        self.vocab += [k for k, v in most_common if v >= self.min_freq]
        self.s2i = {s: i for i, s in enumerate(self.vocab)}
        self.fitted = True
        return self

    def transform(self, examples, bos=True, eos=True):
        bos = [self.index(self.bos_token)] if self.bos_token and bos else []
        eos = [self.index(self.eos_token)] if self.eos_token and eos else []
        for example in examples:
            if self.sequential:
                example = bos + [self.index(s) for s in example] + eos
            else:
                example = self.index(example)
            yield example


class Dataset(object):
    def __init__(self, src, trg, dicts, fitted=False):
        """
        Constructs a dataset out of source and target pairs. Examples will
        be transformed into integers according to their respective Dict.
        The required dicts can be references to the same Dict instance in
        case of e.g. monolingual data.

        Arguments:
            src: list of lists of hashables representing source sequences
            trg: list of lists of hashables representing target sequences
            dicts: dict of {'src': src_dict, 'trg': trg_dict} where
                src_dict: Dict instance fitted to the source data
                trg_dict: Dict instance fitted to the target data
            sort_key: function to sort src, trg example pairs
        """
        self.src = src if fitted else list(dicts['src'].transform(src))
        self.trg = trg if fitted else list(dicts['trg'].transform(trg))
        assert len(src) == len(trg), \
            "Source and Target dataset must be equal length"
        self.dicts = dicts      # fitted dicts

    def __len__(self):
        return len(self.src)

    def sort_(self, sort_key=None):
        src, trg = zip(*sorted(zip(self.src, self.trg), key=sort_key))
        self.src, self.trg = src, trg
        return self

    def batches(self, batch_size, **kwargs):
        """
        Returns a BatchIterator built from this dataset examples

        Parameters:
            batch_size: Integer
            kwargs: Parameters passed on to the BatchIterator constructor
        """
        total_len = len(self.src)
        assert batch_size <= total_len, \
            "Batch size larger than data [%d > %d]" % (batch_size, total_len)
        return BatchIterator(self, batch_size, **kwargs)

    def splits(self, dev=0.1, test=0.2, shuffle=False,
               batchify=False, batch_size=None, sort_key=None, **kwargs):
        """
        Compute splits on dataset instance. For convenience, it can return
        BatchIterator objects instead of Dataset via method chaining.

        Parameters:
            dev: float less than 1 or None, dev set proportion
            test: float less than 1 or None, test set proportion
            shuffle: bool, whether to shuffle the datasets prior to splitting
            batchify: bool, whether to return BatchIterator's instead
            batch_size: int, only needed if batchify is True
            kwargs: optional arguments passed to the BatchIterator constructor
        """
        train = 1 - sum(split for split in [dev, test] if split)
        assert train > 0, "dev and test proportions must add to less than 1"
        if shuffle:
            src, trg = shuffle_pairs(self.src, self.trg)
        else:
            src, trg = self.src, self.trg
        splits = cumsum(int(len(src) * i) for i in [train, dev, test] if i)
        datasets = [Dataset(src[i:j], trg[i:j], self.dicts, fitted=True) \
                    .sort_(sort_key) for i, j in zip(splits, splits[1:])]
        if batchify:
            return tuple([s.batches(batch_size, **kwargs) for s in datasets])
        else:
            return datasets

    @classmethod
    def from_disk(cls, path):
        data = torch.load(path)
        src, trg, dicts = data['src'], data['trg'], data['dicts']
        return cls(src, trg, dicts, fitted=True)

    def to_disk(self, path):
        data = {'src': self.src, 'trg': self.trg, 'dicts': self.dicts}
        torch.save(data, path)


class BatchIterator(object):
    def __init__(self, dataset, batch_size,
                 gpu=False, align_right=False, evaluation=False):
        """
        BatchIterator
        """
        self.dataset = dataset
        self.src = dataset.src
        self.trg = dataset.trg
        self.batch_size = batch_size
        self.gpu = gpu
        self.align_right = align_right
        self.evaluation = evaluation
        self.num_batches = len(dataset) // batch_size

    def _batchify(self, batch_data, pad_token):
        out = batchify(batch_data, pad_token, align_right=self.align_right)
        if self.gpu:
            out = out.cuda()
        return torch.autograd.Variable(out, volatile=self.evaluation)

    def __getitem__(self, idx):
        assert idx < self.num_batches, "%d >= %d" % (idx, self.num_batches)
        batch_from = idx * self.batch_size
        batch_to = (idx+1)*self.batch_size
        src_pad = self.dataset.dicts['src'].get_pad()
        trg_pad = self.dataset.dicts['trg'].get_pad()
        src_batch = self._batchify(self.src[batch_from: batch_to], src_pad)
        trg_batch = self._batchify(self.trg[batch_from: batch_to], trg_pad)
        return src_batch, trg_batch

    def __len__(self):
        return self.num_batches


if __name__ == '__main__':
    # load file parse it and store to file
    pass
