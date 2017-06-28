
import logging
import random
from collections import Counter, Sequence
from functools import singledispatch

import torch
import torch.utils.data
from torch.autograd import Variable


def shuffle_pairs(pair1, pair2):
    for i in reversed(range(1, len(pair1))):
        j = int(random.random() * (i + 1))
        pair1[i], pair1[j] = pair1[j], pair1[i]
        pair2[i], pair2[j] = pair2[j], pair2[i]


def argsort(seq, reverse=False):
    return sorted(range(len(seq)), reverse=reverse, key=seq.__getitem__)


def cumsum(seq):
    seq = [0] + list(seq)
    subseqs = (seq[:i] for i in range(1, len(seq)+1))
    return [sum(subseq) for subseq in subseqs]


def get_splits(length, test, dev=None):
    splits = [split for split in [dev, test] if split]
    train = 1 - sum(splits)
    assert train > 0, "dev and test proportions must add to at most 1"
    return cumsum(int(length * i) for i in [train, dev, test] if i)


def _pack_simple(examples, pad_token=None, align_right=False):
    if pad_token is None:
        assert all(len(examples[0]) == len(x) for x in examples), \
            "No pad token was supported but need to pad (unequal lengths)"
    max_length = max(len(x) for x in examples)
    out = torch.LongTensor(len(examples), max_length).fill_(pad_token or 0)
    for i in range(len(examples)):
        example = torch.Tensor(examples[i])
        example_length = example.size(0)
        offset = max_length - example_length if align_right else 0
        out[i].narrow(0, offset, example_length).copy_(example)
    out = out.t().contiguous()
    return out


def pack(examples, pad_token=None, align_right=False):
    if not isinstance(examples[0], tuple):  # normal input
        return _pack_simple(examples, pad_token, align_right)
    else:                       # multi-input sequences
        return tuple(_pack_simple(e, pad_token, align_right)
                     for e in zip(*examples))


@singledispatch
def wrap_variable(out, volatile, gpu):
    out = Variable(out, volatile=volatile)
    if gpu:
        out = out.cuda()
    return out


@wrap_variable.register(tuple)
def _wrap_variable(out, volatile, gpu):
    return tuple(wrap_variable(subout, volatile, gpu) for subout in out)


def default_sort_key(pair):
    src, trg = pair
    if isinstance(src, tuple):
        return len(src[0])
    return len(src)


@singledispatch
def get_dict_pad(d):
    return None if not d.sequential else d.get_pad()


@get_dict_pad.register(tuple)
def get_dict_pad_(d):
    return next(get_dict_pad(subd) for subd in d)


def _block_batchify(vector, batch_size):
    if isinstance(vector, list):
        vector = torch.LongTensor(vector)
    num_batches = len(vector) // batch_size
    batches = vector.narrow(0, 0, num_batches * batch_size)
    batches = batches.view(batch_size, -1).t().contiguous()
    return batches


def block_batchify(vector, batch_size):
    if isinstance(vector, tuple):
        return tuple(_block_batchify(v, batch_size) for v in vector)
    return _block_batchify(vector, batch_size)


def _debatchify(t):
    return t.t().contiguous().view(-1)


def debatchify(block_batchified):
    if isinstance(block_batchified, tuple):
        return tuple(_debatchify(b) for b in block_batchified)
    return _debatchify(block_batchified)


class Dict(object):
    """
    Dict class to vectorize discrete data.

    Parameters:
    ===========
    - pad_token: None or str, symbol for representing padding
    - eos_token: None or str, symbol for representing end of line
    - bos_token: None or str, symbol for representing begining of line
    - unk_token: None or str, symbol for representing unknown tokens
    - force_unk: bool, Whether to force the inclusion of the unknown symbol
    - max_size: None or int, Maximum size of the dictionary
    - min_freq: int, Minimum freq for a symbol to be included in the dict
    - sequential: bool, Whether the data is sequential (this will entail
        that eos_token and bos_token will be added to examples, unless
        they are None).
    """
    def __init__(self, pad_token=None, eos_token=None, bos_token=None,
                 unk_token='<unk>', force_unk=True, max_size=None, min_freq=1,
                 sequential=True):
        self.counter = Counter()
        self.fitted = False
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        # only index unk_token if needed or requested
        self.reserved = {t for t in [pad_token, eos_token, bos_token] if t}
        if force_unk:
            assert unk_token is not None, "<unk> token needed"
            self.reserved.add(self.unk_token)
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

    def index(self, s):
        if s not in self.s2i and self.unk_token not in self.s2i:
            raise ValueError("OOV [%s, type: %s] with no unk code" %
                             (str(s), str(type(s))))
        return self.s2i.get(s, self.get_unk())

    def partial_fit(self, *datasets):
        for dataset in datasets:
            for example in dataset:
                self.counter.update(example)  # example should always be a list

    def fit(self, *datasets):
        if self.fitted:
            raise ValueError('Dict is already fitted')
        self.partial_fit(*datasets)
        most_common = self.counter.most_common(self.max_size)
        if self.max_size is not None and self.max_size < len(self.counter):
            if self.unk_token and self.unk_token not in self.reserved:
                self.reserved.add(self.unk_token)
        self.vocab = [s for s in self.reserved]
        self.vocab += [k for k, v in most_common if v >= self.min_freq]
        self.s2i = {s: i for i, s in enumerate(self.vocab)}
        self.fitted = True
        return self

    def transform(self, examples, bos=True, eos=True):
        assert self.fitted, "Attempt to index without fitted data"
        bos = [self.get_bos() if self.bos_token and bos else []]
        eos = [self.get_eos() if self.eos_token and eos else []]
        for example in examples:
            if self.sequential:
                yield bos + [self.index(s) for s in example] + eos
            else:
                yield [self.index(s) for s in example]


class Dataset(Sequence, torch.utils.data.Dataset):
    """
    Abstract class wrapping torch.utils.data.Dataset
    """
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_disk(cls, path):
        with open(path, 'rb') as f:
            return torch.load(f)

    def to_disk(self, path):
        with open(path, 'wb') as f:
            torch.save(self, f)


class PairedDataset(Dataset):
    def __init__(self, src, trg, d, batch_size=1,
                 fitted=False, gpu=False, evaluation=False, align_right=False):
        """
        Constructs a dataset out of source and target pairs. Examples will
        be transformed into integers according to their respective Dict.
        The required dicts can be references to the same Dict instance in
        case of e.g. monolingual data.

        Arguments:
        - src: (list or tuple) of lists of lists of hashables representing
            source sequences. If a tuple, each member should be a parallel
            version of the same sequence.
        - trg: same as src but repersenting target sequences.
        - d: dict of {'src': src_dict, 'trg': trg_dict} where
            src_dict: Dict or tuple of Dicts fitted to the source data. If
                passed a list, the Dicts should be order to match the order
                of the parallel version passed to src
            trg_dict: same as src_dict but for the target data
        """
        self.autoregressive = False
        self.data, self.d = {}, d
        self.data['src'] = src if fitted else self._fit(src, d['src'])
        if trg is None:         # autoregressive dataset
            self.autoregressive = True
            self.data['trg'] = self.data['src']
            self.d['trg'] = self.d['src']
        else:
            self.data['trg'] = trg if fitted else self._fit(trg, d['trg'])
            assert len(self.data['src']) == len(self.data['trg']), \
                "source and target dataset must be equal length"
        assert len(self.data['src']) >= batch_size, "not enough input examples"
        self.d = d              # fitted dicts
        self.batch_size = batch_size
        self.gpu = gpu
        self.evaluation = evaluation
        self.align_right = align_right
        self.num_batches = len(self.data['src']) // batch_size

    def _fit(self, data, d):
        # check inputs
        if isinstance(data, tuple) or isinstance(d, tuple):
            assert isinstance(data, tuple) and isinstance(d, tuple), \
                "both input sequences and Dict must be equal type"
            assert all(len(data[i]) == len(data[i+1])
                       for i in range(len(data)-2)), \
                "all input datasets must be equal size"
            assert len(data) == len(d), \
                "equal number of input sequences and Dicts needed"
            return list(zip(*(subd.transform(subdata)
                              for subdata, subd in zip(data, d))))
        else:
            return list(d.transform(data))

    def _pack(self, batch_data, pad_token=None):
        out = pack(
            batch_data, pad_token=pad_token, align_right=self.align_right)
        return wrap_variable(out, volatile=self.evaluation, gpu=self.gpu)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        assert idx < self.num_batches, "%d >= %d" % (idx, self.num_batches)
        b_from, b_to = idx * self.batch_size, (idx+1) * self.batch_size
        src_pad = get_dict_pad(self.d['src'])
        src = self._pack(self.data['src'][b_from: b_to], pad_token=src_pad)
        trg_pad = get_dict_pad(self.d['trg'])
        trg = self._pack(self.data['trg'][b_from: b_to], pad_token=trg_pad)
        return src, trg

    def set_batch_size(self, new_batch_size):
        if self.batch_size == new_batch_size:
            return
        self.batch_size = new_batch_size
        self.num_batches = len(self.data['src']) // new_batch_size

    def set_gpu(self, new_gpu):
        self.gpu = new_gpu

    def sort_(self, sort_by='src', reverse=True):
        """
        Sort dataset examples according to sequence length. By default source
        sequences are used for sorting (see sort_by function).

        Parameters:
        -----------
        sort_by: one of ('src', 'trg'), Sort instances according to the length
            of the source or the target dataset.
        """
        assert sort_by in ('src', 'trg')
        indices = None
        if self.autoregressive:
            if sort_by == 'trg':
                logging.warn("Omitting sort_by in autoregressive dataset")
            self.data['src'].sort(reverse=reverse)
        else:
            index = argsort(list(map(len, self.data[sort_by])), reverse=reverse)
            self.data['src'] = [self.data['src'][idx] for idx in index]
            self.data['trg'] = [self.data['trg'][idx] for idx in index]
        return self

    def splits(self, test=0.1, dev=0.2, shuffle=False, sort_by=None):
        """
        Compute splits on dataset instance. For convenience, it can return
        BatchIterator objects instead of Dataset via method chaining.

        Parameters:
        ===========
        - dev: float less than 1 or None, dev set proportion
        - test: float less than 1 or None, test set proportion
        - shuffle: bool, whether to shuffle the datasets prior to splitting
        """
        if shuffle:
            if self.autoregressive:
                random.shuffle(self.data['src'])
            else:
                shuffle_pairs(self.data['src'], self.data['trg'])
        splits = get_splits(len(self.data['src']), test, dev=dev)
        datasets = []
        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = self.evaluation if idx == 0 else True
            if self.autoregressive:
                subset = PairedDataset(
                    self.data['src'][start:stop], None,
                    self.d, self.batch_size, fitted=True, gpu=self.gpu,
                    evaluation=evaluation, align_right=self.align_right)
            else:
                subset = PairedDataset(
                    self.data['src'][start:stop], self.data['trg'][start:stop],
                    self.d, self.batch_size, fitted=True, gpu=self.gpu,
                    evaluation=evaluation, align_right=self.align_right)
            if sort_by:
                subset.sort_(sort_by=sort_by)
            datasets.append(subset)
        return tuple(datasets)


class BlockDataset(Dataset):
    """
    Dataset class for autoregressive models.

    Parameters:
    ===========
    - examples: list of source sequences where each input sequence is a list,
        Multiple input can be used by passing a tuple of input sequences.
        In the latter case the each batch will be a tuple with entries
        corresponding to the different domains in the same order as passed
        to the constructor, and the input to d must be a tuple in which each
        entry is a Dict fitted to the corresponding input domain.
        If fitted is False, the lists are supposed to be already transformed
        into a single long vector.
    - d: Dict (or tuple of Dicts for multi-input) already fitted.
    - batch_size: int,
    - bptt: int,
        Backprop through time (max context the RNN conditions predictions on)
    """
    def __init__(self, examples, d, batch_size, bptt,
                 fitted=False, gpu=False, evaluation=False):
        if not fitted:
            examples = self._fit(examples, d, batch_size)
        self.data = block_batchify(examples, batch_size)
        self.d = d
        self.batch_size = batch_size
        self.bptt = bptt
        self.fitted = fitted
        self.gpu = gpu
        self.evaluation = evaluation

    def _fit(self, examples, d, batch_size):
        too_short = ValueError(
            "Not enough data for a batch size [%d]" % batch_size)
        if isinstance(examples, tuple) or isinstance(d, tuple):
            assert isinstance(d, tuple) and isinstance(examples, tuple), \
                "multiple input needs multiple Dicts"
            assert len(examples) == len(d), \
                "equal number of input and Dicts needed"
            assert all(len(examples[i]) == len(examples[i+1])
                       for i in range(len(examples)-2)), \
                "all input examples must be equal size"
            if len(examples[0]) // batch_size == 0:
                raise too_short
            return tuple(
                [item for seq in subd.transform(subex) for item in seq]
                for (subex, subd) in zip(examples, d))
        else:
            if len(examples) // batch_size == 0:
                raise too_short
            return [item for seq in d.transform(examples) for item in seq]

    def _getitem(self, data, idx):
        """
        General function to get the source data to compute the batch. This
        should be overwritten by subclasses in which the source data isn't
        always stored in self.data, e.g. the case of cyclical subset access.
        """
        idx *= self.bptt
        seq_len = min(self.bptt, len(data) - 1 - idx)
        src = Variable(data[idx:idx+seq_len], volatile=self.evaluation)
        trg = Variable(data[idx+1:idx+seq_len+1], volatile=self.evaluation)
        if self.gpu:
            src, trg = src.cuda(), trg.cuda()
        return src, trg

    def __len__(self):
        """
        The length of the dataset is computed as the number of bptt'ed batches
        to conform the way batches are computed. See __getitem__.
        """
        if isinstance(self.data, tuple):
            return len(self.data[0]) // self.bptt
        return len(self.data) // self.bptt

    def __getitem__(self, idx):
        """
        Item getter for batch number idx.

        Returns:
        ========
        - src: torch.LongTensor of maximum size self.bptt x self.batch_size
        - trg: torch.LongTensor of maximum size self.bptt x self.batch_size,
            corresponding to a shifted batch
        """
        if isinstance(self.data, tuple):
            return tuple(zip(*(self._getitem(d, idx) for d in self.data)))
        else:
            return self._getitem(self.data, idx)

    def set_gpu(self, new_gpu):
        self.gpu = new_gpu

    def set_batch_size(self, new_batch_size):
        if self.batch_size == new_batch_size:
            return
        self.batch_size = new_batch_size
        self.data = block_batchify(debatchify(self.data), new_batch_size)

    def split_data(self, start, stop):
        """
        Compute a split on the dataset for a batch range defined by start, stop
        """
        if isinstance(self.data, tuple):
            return tuple(d.t().contiguous().view(-1)[start:stop]
                         for d in self.data)
        else:
            return self.data.t().contiguous().view(-1)[start:stop]

    def splits(self, test=0.1, dev=0.1):
        """
        Computes splits according to test and dev proportions (whose sum can't
        be higher than 1). In case of a multi-source dataset, the output is
        respectively a dataset containing the partition for each source in the
        same shape as the original (non-partitioned) dataset.

        Returns:
        ========

        tuple of BlockDataset's
        """
        n_element = len(self) * self.bptt * self.batch_size  # min num elements
        datasets, splits = [], get_splits(n_element, test, dev=dev)
        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = self.evaluation if idx == 0 else True
            datasets.append(type(self)(
                self.split_data(start, stop), self.d, self.batch_size,
                self.bptt, fitted=True, gpu=self.gpu, evaluation=evaluation))
        return tuple(datasets)

    @classmethod
    def splits_from_data(cls, data, d, batch_size, bptt,
                         gpu=False, test=0.1, dev=0.1, evaluation=False):
        """
        Shortcut classmethod for loading splits from a vector-serialized
        corpus. It can be used to avoid creating the parent dataset before
        creating the children splits if the data is already in vector form.
        """
        datasets, splits = [], get_splits(len(data), test, dev=dev)
        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = evaluation if idx == 0 else True
            datasets.append(cls(data[start:stop], d, batch_size, bptt,
                                fitted=True, gpu=gpu, evaluation=evaluation))
        return tuple(datasets)


class CyclicBlockDataset(BlockDataset):
    def __init__(self, examples, d, batch_size, bptt,
                 fitted=False, gpu=False, evaluation=False):
        self.data = {}
        for name, data in examples.items():
            if not fitted:      # subdata is already an integer vector
                data = [c for l in d.transform(data) for c in l]
            self.data[name] = block_batchify(data, batch_size)

        self.names = list(self.data.keys())
        self.d = d
        self.batch_size = batch_size
        self.bptt = bptt
        self.fitted = fitted
        self.gpu = gpu
        self.evaluation = evaluation

    def __len__(self):
        batches = min(len(d) for d in self.data.values()) * len(self.data)
        return batches // self.bptt

    def __getitem__(self, idx):
        """
        Computes the subset for each batch in a cyclical way.

        Returns:
        ========
        In addition to the standard BlockDataset, this subclass also returns
        the name of the dataset in the current cycle in third position.
        """
        idx, dataset = divmod(idx, len(self.data))
        name = self.names[dataset]
        data = self.data[name]
        src, trg = self._getitem(data, idx)
        return src, trg, name

    def split_data(self, start, stop):
        start, stop = start // len(self.data), stop // len(self.data)
        split = {}
        for name, data in self.data.items():
            split[name] = data.t().contiguous().view(-1)[start:stop]
        return split
