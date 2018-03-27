
import math
import logging
import random
from collections import Counter, Sequence, OrderedDict, defaultdict

import torch
import torch.utils.data

from seqmod.utils import UNK, wrap_variables


def bucketing(*args):
    """
    - args: sequence of ints in increasing order that define buckets
    """
    args = sorted(args)

    def func(x):
        x = int(x)
        for idx, arg in enumerate(args):
            if x < arg:
                return str(idx)
        return str(len(args))

    return func


def truncate(max_size):
    """
    - max_size: int, maximum size to truncate to.
    """
    if max_size is None:
        def func(x): return x

    else:

        def func(x):
            if len(x) <= max_size:
                return x
            return x[:max_size]

    return func


def shuffle_pairs(pair1, pair2):
    """
    Symmetrically shuffle two lists
    """
    for i in reversed(range(1, len(pair1))):
        j = int(random.random() * (i + 1))
        pair1[i], pair1[j] = pair1[j], pair1[i]
        pair2[i], pair2[j] = pair2[j], pair2[i]


def argsort(seq, reverse=False):
    """
    Compute the indices that sort a sequence
    """
    return sorted(range(len(seq)), reverse=reverse, key=seq.__getitem__)


def destruct(tup, idx):
    """
    Destruct a tuple into three tuples corresponding to the prefix,
    the actual target value `idx`, and the suffix (which might be empty)
    if the idx refers to the last tuple value
    """
    if idx >= len(tup):
        raise IndexError("Index out of range")
    if idx == 0:
        return (), tup[0:1], tup[1:]
    else:
        return tup[0:idx], tup[idx], tup[idx+1:]


def cumsum(seq):
    """
    Compute the cumulative sum of an input sequence

    >>> cumsum([1, 2, 3, 4])
    [0, 1, 3, 6, 10]
    """
    seq = [0] + list(seq)
    subseqs = (seq[:i] for i in range(1, len(seq) + 1))
    return [sum(subseq) for subseq in subseqs]


def get_splits(length, test, dev=None):
    """
    Compute splits for a given dataset length
    """
    splits = [split for split in [dev, test] if split]
    train = 1 - sum(splits)
    if train > 1 or train < 0:
        raise ValueError("Illegal proportions test: {}, dev: {}"
                         .format(test, dev))
    return cumsum(int(length * i) for i in [train, dev, test] if i)


def pad_batch(examples, pad, return_lengths, align_right):
    """
    Transform a list of examples into a proper torch.LongTensor batch
    """
    lengths = list(map(len, examples))

    if pad is None and not all(lengths[0] == l for l in lengths[1:]):
        raise ValueError("Variable length without padding")

    # create batch
    maxlen = max(lengths)
    out = torch.LongTensor(len(examples), maxlen).fill_(pad or 0)
    for i, example in enumerate(examples):
        left = 0 if not align_right else maxlen - len(example)
        out[i].narrow(0, left, len(example)).copy_(torch.Tensor(example))

    # turn to batch second
    out = out.t().contiguous()

    if return_lengths:
        return out, lengths
    else:
        return out


def block_batchify(vector, batch_size):
    """
    Transform input vector to (None, batch_size)

    >>> block_batchify([0, 2, 4, 6, 1, 3, 5, 7], 2).tolist()
    [[0, 1], [2, 3], [4, 5], [6, 7]]
    """
    if isinstance(vector, tuple):
        return tuple(block_batchify(v, batch_size) for v in vector)

    if isinstance(vector, list):
        vector = torch.LongTensor(vector)

    length = (len(vector) // batch_size) * batch_size

    return vector.narrow(0, 0, length).view(batch_size, -1).t().contiguous()


def debatchify(t):
    """
    Reverse operation to block_batchify
    """
    if isinstance(t, tuple):
        return tuple(debatchify(subt) for subt in t)
    return t.t().contiguous().view(-1)


class Dict(object):
    """
    Dict class to vectorize discrete data.

    Parameters:
    -----------
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
                 unk_token=UNK, force_unk=False, max_size=None, min_freq=1,
                 sequential=True, max_len=None, use_vocab=True, dtype=int,
                 preprocessing=None):

        self.counter = Counter()
        self.vocab = []
        self.s2i = {}
        self.reserved = set()
        self.fitted = False
        self.use_vocab = use_vocab
        self.dtype = dtype
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unk_token = unk_token
        self.preprocessing = preprocessing

        # only index unk_token if needed or requested
        if self.use_vocab:
            self.reserved = {t for t in [pad_token, eos_token, bos_token] if t}
            if force_unk or sequential:  # force unk for sequential dicts
                if unk_token is None:
                    raise ValueError("<unk> token needed")
                self.reserved.add(self.unk_token)

        self.max_size = max_size
        self.min_freq = min_freq
        self.max_len = max_len
        self.sequential = sequential

    def __repr__(self):
        rep = ("<Dict fitted={}, max_size={}, min_freq={}, bos_token={}, "
               "eos_token={}, unk_token={}, sequential={}").format(
                   self.fitted, self.max_size, self.min_freq, self.bos_token,
                   self.eos_token, self.unk_token, self.sequential)
        if self.fitted:
            rep += ' n_symbols={}>'.format(len(self))
        else:
            rep += '>'
        return rep

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
            raise ValueError("OOV {} of type {} but no UNK".format(s, type(s)))

        return self.s2i.get(s, self.get_unk())

    def partial_fit(self, *datasets):
        for dataset in datasets:
            if not self.sequential:
                if self.preprocessing is not None:
                    self.counter.update(self.preprocessing(ex) for ex in dataset)
                else:
                    self.counter.update(dataset)
            else:
                for ex in dataset:
                    if self.max_len is not None and len(ex) > self.max_len:
                        ex = ex[:self.max_len]
                    if self.preprocessing is not None:
                        ex = self.preprocessing(ex)
                    self.counter.update(ex)

    def fit(self, *datasets):
        """
        Parameters:
        -----------
        - datasets: one or more datasets consisting of lists of examples.
            Each example will be either an iterable if sequential or a
            hashble.
        """
        if self.fitted:
            raise ValueError('Dict is already fitted')

        if not self.use_vocab:
            logging.warn("Dict is setup to not use a vocabulary. `fit` will be ignored")
            return self

        self.partial_fit(*datasets)
        self.compute_vocab()

        return self

    def compute_vocab(self):
        most_common = self.counter.most_common(self.max_size)

        # add unk token to vocabulary if needed
        if self.max_size is not None and self.max_size < len(self.counter):
            if self.unk_token and self.unk_token not in self.reserved:
                self.reserved.add(self.unk_token)

        # create tables (vocab is kept in sorted order which is needed for
        # efficient candidate sampling; reserved symbols are added as most
        # frequent at the top of the list)
        self.vocab = [s for s in self.reserved] + \
                     [k for k, v in most_common if v >= self.min_freq]
        self.s2i = {s: i for i, s in enumerate(self.vocab)}
        self.fitted = True

    def transform(self, examples):
        """
        Parameters
        ----------
        - examples: list of examples. An example is either an iterable if
            sequential or a hashable.
        """
        if not self.fitted and self.use_vocab:
            raise ValueError("Attempt to index without fitted data")

        # only for sequential models
        bos = [self.get_bos()] if self.bos_token else []
        eos = [self.get_eos()] if self.eos_token else []

        for example in examples:
            # preprocess if needed
            if self.preprocessing is not None:
                example = self.preprocessing(example)

            if self.sequential:
                if self.max_len is not None and len(example) > self.max_len:
                    example = example[:self.max_len]
                if self.use_vocab:
                    example = [self.index(s) for s in example]
                yield bos + example + eos
            else:
                if self.use_vocab:
                    yield self.index(example)
                else:
                    yield example

    def pack(self, batch_data, return_lengths=False, align_right=False):
        """
        Convert transformed data into torch batch. Output type is LongTensor.
        This could be adapted to return other types as well.

        Parameter:
        ----------
        - batch_data: a list of examples
        - return_lengths: bool, if True output will be a tuple of LongTensor
            and list with sequence lengths in the batch
        """
        if self.sequential:
            return pad_batch(
                batch_data, self.get_pad(), return_lengths, align_right)
        else:
            if self.dtype is int:
                return torch.LongTensor(batch_data)
            else:
                return torch.FloatTensor(batch_data)


class MultiDict(object):
    """
    Composition Dict to hold multiple dicts together and make easier
    handling multiple input datasets

    Parameters:
    -----------

    - definitions: dict of input key and corresponding Dict parameters
    """
    def __init__(self, definitions):
        self.dicts = OrderedDict()
        for k, kwargs in definitions.items():
            self.dicts[k] = Dict(**kwargs)
        self.fitted = False

    def fit(self, *datasets):
        """
        Parameters
        ----------

        - datasets: one or more input datasets, where each dataset is an
            iterable of dicts with keys agreeing with dict keys
        """
        if self.fitted:
            raise ValueError('Dict is already fitted')

        for dataset in datasets:
            for row in dataset:
                for k, d in self.dicts.items():
                    d.partial_fit([row[k]])

        for d in self.dicts.values():
            d.compute_vocab()

        self.fitted = True

        return self

    def transform(self, examples):
        """
        Transform multi-input examples.

        Parameters
        ----------

        - examples: iterable of dicts with keys agreeing with dict keys
        """
        fitted = [list() for _ in range(len(self.dicts))]
        for seq in examples:
            for idx, (k, d) in enumerate(self.dicts.items()):
                if d.use_vocab:
                    output = next(d.transform([seq[k]]))
                    fitted[idx].append(output)
                else:
                    fitted[idx].append(seq[k])
        return fitted


class CompressionTable(object):
    """
    Simple implementation of a compression mechanism to map input tuples
    to single integers and back.

    Parameters:
    -----------

    - nvals: int, expected number of integers in the input tuple
    """
    def __init__(self, nvals):
        self.index2vals = []
        self.vals2index = {}
        self.nvals = nvals

    def hash_vals(self, vals):
        if len(vals) != self.nvals:
            raise ValueError("Wrong number of values {}".format(len(vals)))
        if vals in self.vals2index:
            return self.vals2index[vals]
        else:
            idx = len(self.vals2index)
            self.index2vals.append(vals)
            self.vals2index[vals] = idx
            return idx

    def get_vals(self, index):
        if index >= len(self.index2vals):
            raise ValueError("Unknown input index [{}]".format(index))
        return self.index2vals[index]

    def expand(self, t):
        """
        Transform a 2D input tensor into `nvals` tensors of same dimensionality
        as the input tensor applying the learned compression to each entry
        """
        seq_len, batch_size = t.size()
        vals1d = (c for i in t.view(-1) for c in self.get_vals(i))
        t = torch.LongTensor(list(vals1d))
        return tuple(t.view(seq_len, batch_size, self.nvals)
                      .transpose(2, 0)  # (nvals, batch_size, seq_len)
                      .transpose(1, 2)  # (nvals, seq_len, batch_size)
                      .contiguous())


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
    """
    Constructs a dataset out of source and target pairs. Examples will
    be transformed into integers according to their respective Dict.
    The required dicts can be references to the same Dict instance in
    case of e.g. monolingual data.

    Parameters
    ----------

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
    def __init__(self, src, trg, d, batch_size=1,
                 fitted=False, gpu=False, evaluation=False,
                 return_lengths=True, align_right=False):
        self.autoregressive, self.data, self.d = False, {}, d

        # prepare src data
        self.data['src'] = src if fitted else self._fit(src, self.d['src'])
        src_len = len(self.data['src'])
        if src_len < batch_size:
            raise ValueError(
                "Not enough input examples for selected batch_size. "
                "Got {} but batch_size is {}".format(src_len, batch_size))

        # prepare trg data
        if trg is None:         # autoregressive dataset
            self.autoregressive = True
            self.data['trg'], self.d['trg'] = self.data['src'], self.d['src']
        else:
            self.data['trg'] = trg if fitted else self._fit(trg, self.d['trg'])
            trg_len = len(self.data['trg'])
            if src_len != trg_len:
                raise ValueError("Source and target must be equal length. Got "
                                 "src {} and trg {}".format(src_len, trg_len))

        self.batch_size = batch_size
        self.gpu = gpu
        self.evaluation = evaluation
        self.return_lengths = return_lengths
        self.align_right = align_right
        self.num_batches = src_len // batch_size

    def _fit(self, data, dicts):
        # multiple input dataset with MultiDict
        if isinstance(dicts, MultiDict):
            return list(zip(*dicts.transform(data)))

        # multiple input dataset
        elif isinstance(data, tuple) or isinstance(dicts, tuple):
            assert isinstance(data, tuple) and isinstance(dicts, tuple), \
                "Input sequence is type {}, but dict is {}".format(
                    type(data), type(dicts))
            assert all(len(data[i]) == len(data[i+1])
                       for i in range(len(data)-2)), \
                "All input datasets must be equal size"
            assert len(data) == len(dicts), \
                "Equal number of input sequences and Dicts needed"
            fitted = (d.transform(subset) for subset, d in zip(data, dicts))
            return list(zip(*fitted))

        # single input
        else:
            return list(dicts.transform(data))

    def _pack(self, batch, dicts):
        # multi-input dataset
        if isinstance(batch[0], tuple):
            batches = list(zip(*batch))  # unpack batches
            if isinstance(dicts, MultiDict):
                dicts = dicts.dicts.values()
            out = tuple(d.pack(b, return_lengths=self.return_lengths,
                               align_right=self.align_right)
                        for (d, b) in zip(dicts, batches))
        else:
            out = dicts.pack(batch, return_lengths=self.return_lengths,
                             align_right=self.align_right)

        return wrap_variables(out, volatile=self.evaluation, gpu=self.gpu)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError("{} >= {}".format(idx, self.num_batches))

        b_from, b_to = idx * self.batch_size, (idx+1) * self.batch_size
        src = self._pack(self.data['src'][b_from: b_to], self.d['src'])
        trg = self._pack(self.data['trg'][b_from: b_to], self.d['trg'])
        return src, trg

    def set_batch_size(self, new_batch_size):
        if self.batch_size == new_batch_size:
            return
        self.batch_size = new_batch_size
        self.num_batches = len(self.data['src']) // new_batch_size

    def set_gpu(self, new_gpu):
        self.gpu = new_gpu

    def sort_(self, key=lambda x: len(x), reverse=True, sort_by='src'):
        """
        Sort dataset examples according to sequence length. By default source
        sequences are used for sorting (see sort_by function).

        Parameters:
        -----------
        sort_by: one of ('src', 'trg'), Sort instances according to the length
            of the source or the target dataset.
        """
        if sort_by not in ('src', 'trg'):
            raise ValueError("sort_by must be one of 'src', 'trg'")

        if self.autoregressive:
            if sort_by == 'trg':
                logging.warn("Omitting sort_by in autoregressive dataset")
            self.data['src'].sort(key=key, reverse=reverse)
        else:
            ix = argsort([key(i) for i in self.data[sort_by]], reverse=reverse)
            self.data['src'] = [self.data['src'][i] for i in ix]
            self.data['trg'] = [self.data['trg'][i] for i in ix]

        return self

    def stratify_(self, target='trg', key=lambda data: data):
        """
        Force balanced batch data according to an input field.

        Parameters:
        -----------
        - target: str, either 'src' or 'trg' to take as reference
        - key: in case of multi-input data a function is needed to retrieve
            the actual target.
        """
        if self.autoregressive:
            logging.warn("Omitting `target` value in autoregressive dataset")
            target = 'src'
        if target not in ('src', 'trg'):
            raise ValueError('`target` must be "src" or "trg"')
        if isinstance(self.data[target], tuple):
            if key is None:
                raise ValueError('Got multi-input dataset but no input `key`')

        # group labels by value
        data, grouped = key(self.data[target]), defaultdict(list)
        for idx, val in enumerate(data):
            grouped[val].append(idx)

        total = len(data)
        probs = {key: len(vals) / total for key, vals in grouped.items()}

        # stratify according to distribution to generate index
        index = []
        for _ in range(0, total, self.batch_size):
            batch = []
            for key, prob in probs.items():
                n, rest = divmod(prob * self.batch_size, 1)
                if random.random() <= rest:
                    n += 1
                for _ in range(int(n)):
                    try:
                        batch.append(grouped[key].pop())
                    except:
                        continue
            index.extend(batch)
        for _, vals in grouped.items():  # remaining items
            index.extend(vals)

        # reorder according to sampled index
        self.data['src'] = [self.data['src'][i] for i in index]
        if not self.autoregressive:
            self.data['trg'] = [self.data['trg'][i] for i in index]

        return self

    def shuffle_(self):
        """Shuffle underlying data keeping the src to trg pairings"""
        if self.autoregressive:
            random.shuffle(self.data['src'])
        else:
            shuffle_pairs(self.data['src'], self.data['trg'])

        return self

    def splits(self, test=0.1, dev=0.2, shuffle=False, sort=False, **kwargs):
        """
        Compute splits on dataset instance. For convenience, it can return
        BatchIterator objects instead of Dataset via method chaining.

        Parameters
        ----------

        - dev: float less than 1 or None, dev set proportion
        - test: float less than 1 or None, test set proportion
        - shuffle: bool, whether to shuffle the datasets prior to splitting
        """
        if shuffle:
            self.shuffle_()

        splits, sets = get_splits(len(self.data['src']), test, dev=dev), []

        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = self.evaluation if idx == 0 else True
            # get dataset splits
            if self.autoregressive:
                src, trg = self.data['src'][start:stop], None
            else:
                src = self.data['src'][start:stop]
                trg = self.data['trg'][start:stop]

            subset = type(self)(
                src, trg, self.d, self.batch_size, fitted=True, gpu=self.gpu,
                return_lengths=self.return_lengths, evaluation=evaluation,
                align_right=self.align_right)

            if sort:
                subset.sort_(**kwargs)

            sets.append(subset)

        return tuple(sets)


class BlockDataset(Dataset):
    """
    Dataset class for autoregressive models.

    Parameters
    ----------

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
                 fitted=False, gpu=False, evaluation=False,
                 table=None, table_idx=1):
        if not fitted:
            examples = self._fit(examples, d, batch_size)
        self.data = block_batchify(examples, batch_size)
        self.d = d
        self.batch_size = batch_size
        self.bptt = bptt
        self.fitted = fitted
        self.gpu = gpu
        self.evaluation = evaluation
        self.table = table
        self.table_idx = table_idx

    def _fit(self, examples, dicts, batch_size):
        # multiple input dataset with MultiDict
        if isinstance(dicts, MultiDict):
            fitted = dicts.transform(examples)
            return tuple([i for seq in subset for i in seq] for subset in fitted)

        # multiple input dataset
        if isinstance(examples, tuple) or isinstance(dicts, tuple):
            assert isinstance(dicts, tuple) and isinstance(examples, tuple), \
                "Input sequence is type {}, but dict is {}".format(
                    type(examples), type(dicts))
            assert len(examples) == len(dicts), \
                "Equal number of input sequences and Dicts needed"
            assert all(len(examples[i]) == len(examples[i+1])
                       for i in range(len(examples)-2)), \
                "All input examples must be equal size"
            if len(examples[0]) // batch_size == 0:
                raise ValueError("Not enough data for batch [{}]"
                                 .format(batch_size))

            fitted = [d.transform(e) for d, e in zip(dicts, examples)]
            return tuple([i for seq in subset for i in seq] for subset in fitted)

        # single input dataset
        else:
            if len(examples) // batch_size == 0:
                raise ValueError("Not enough data for batch [{}]"
                                 .format(batch_size))
            return [i for seq in dicts.transform(examples) for i in seq]

    def _get_batch(self, data, idx):
        """
        General function to get the source data to compute the batch. This
        should be overwritten by subclasses in which the source data isn't
        always stored in self.data, e.g. the case of cyclical subset access.
        """
        idx *= self.bptt
        seq_len = min(self.bptt, len(data) - 1 - idx)
        src_data, trg_data = data[idx:idx+seq_len], data[idx+1:idx+seq_len+1]
        src = wrap_variables(src_data, self.evaluation, self.gpu)
        trg = wrap_variables(trg_data, self.evaluation, self.gpu)
        return src, trg

    def __len__(self):
        """
        The length of the dataset is computed as the number of bptt'ed batches
        to conform the way batches are computed. See __getitem__.
        """
        basis = self.data[0] if isinstance(self.data, tuple) else self.data
        # if batches don't divide evenly by bptt there will be an extra last
        # batch with lower bptt
        return math.ceil((len(basis) - 1) / self.bptt)

    def __getitem__(self, idx):
        """
        Item getter for batch number idx.

        Returns
        -------

        - src: torch.LongTensor of maximum size self.bptt x self.batch_size
        - trg: torch.LongTensor of maximum size self.bptt x self.batch_size,
            corresponding to a shifted batch
        """
        src, trg = None, None
        # multi-input
        if isinstance(self.data, tuple):
            src, trg = tuple(zip(*(self._get_batch(d, idx) for d in self.data)))
            # decompress from table
            if self.table is not None:
                # source
                src_pre, src_target, src_post = destruct(src, self.table_idx)
                src_target = tuple(wrap_variables(t, self.evaluation, self.gpu)
                                   for t in self.table.expand(src_target.data))
                src = tuple(src_pre + src_target + src_post)
                # target
                trg_pre, trg_target, trg_post = destruct(trg, self.table_idx)
                trg_target = tuple(wrap_variables(t, self.evaluation, self.gpu)
                                   for t in self.table.expand(trg_target.data))
                trg = tuple(trg_pre + trg_target + trg_post)
        # single-input
        else:
            src, trg = self._get_batch(self.data, idx)
        return src, trg

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

        Returns
        -------

        tuple of BlockDataset's
        """
        n_element = len(self) * self.bptt * self.batch_size  # min num elements
        subsets, splits = [], get_splits(n_element, test, dev=dev)

        table = self.table if hasattr(self, 'table') else None
        table_idx = self.table_idx if hasattr(self, 'table_idx') else None

        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = self.evaluation if idx == 0 else True
            subsets.append(type(self)(
                self.split_data(start, stop), self.d, self.batch_size,
                self.bptt, fitted=True, gpu=self.gpu, evaluation=evaluation,
                table=table, table_idx=table_idx))
        return tuple(subsets)

    @classmethod
    def splits_from_data(cls, data, d, batch_size, bptt, test=0.1, dev=0.1,
                         **kwargs):
        """
        Shortcut classmethod for loading splits from a vector-serialized
        corpus. It can be used to avoid creating the parent dataset before
        creating the children splits if the data is already in vector form.
        """
        size = len(data[0]) if isinstance(data, tuple) else len(data)
        subsets, splits = [], get_splits(size, test, dev=dev)

        for idx, (start, stop) in enumerate(zip(splits, splits[1:])):
            evaluation = False if idx == 0 else True
            if isinstance(data, tuple):
                split = tuple(d[start:stop] for d in data)
            else:
                split = data[start:stop]
            dataset = cls(split, d, batch_size, bptt,
                          fitted=True, evaluation=evaluation, **kwargs)
            subsets.append(dataset)
        return tuple(subsets)


class CyclicBlockDataset(BlockDataset):
    def __init__(self, examples, d, batch_size, bptt,
                 fitted=False, gpu=False, evaluation=False, **kwargs):
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

        Returns
        -------

        In addition to the standard BlockDataset, this subclass also returns
        the name of the dataset in the current cycle in third position.
        """
        idx, dataset = divmod(idx, len(self.data))
        name = self.names[dataset]
        data = self.data[name]
        src, trg = self._get_batch(data, idx)
        return src, trg, name

    def split_data(self, start, stop):
        start, stop = start // len(self.data), stop // len(self.data)
        split = {}
        for name, data in self.data.items():
            split[name] = data.t().contiguous().view(-1)[start:stop]
        return split
