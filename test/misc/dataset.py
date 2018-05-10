
import unittest
from hashlib import sha1, md5
from collections import Counter, defaultdict

import lorem
import numpy as np
import torch

from seqmod.misc import Dict, BlockDataset, PairedDataset, CompressionTable
from seqmod.misc.dataset import argsort
from seqmod import utils as u


def fake_tags(w):
    return (w, md5(w.encode('utf-8')), sha1(w.encode('utf-8')))


class TestDict(unittest.TestCase):
    def setUp(self):
        self.corpus = [lorem.sentence().split() for _ in range(100)]
        self.seq_vocab = Counter(w for s in self.corpus for w in s)
        self.seq_d = Dict(eos_token=u.EOS, bos_token=u.BOS,
                          force_unk=True, sequential=True)
        self.seq_d.fit(self.corpus)
        self.seq_transformed = list(self.seq_d.transform(self.corpus))

    def test_vocab(self):
        self.assertEqual(
            len(self.seq_d), len(self.seq_vocab) + 3,
            "Vocabulary matches for a Dict with padding and bos/eos tokens")
        diff = set(self.seq_d.vocab) - set(self.seq_vocab.keys())
        self.assertEqual(
            diff, set([u.EOS, u.BOS, u.UNK]),
            "Entries in the vocabulary matches")

    def test_transform(self):
        self.assertEqual(
            self.corpus,
            # remove <bos>, <eos> from transformed corpus
            [[self.seq_d.vocab[w] for w in s[1:-1]]
             for s in self.seq_d.transform(self.corpus)],
            "Transformed corpus matches word by word")


class TestBlockDataset(unittest.TestCase):
    def setUp(self):
        self.corpus = [lorem.sentence().split() for _ in range(100)]
        self.tagged_corpus = [[fake_tags(w) for w in s] for s in self.corpus]
        self.tag1_corpus = [[tup[1] for tup in s] for s in self.tagged_corpus]
        self.tag2_corpus = [[tup[2] for tup in s] for s in self.tagged_corpus]
        # dicts
        self.seq_d = Dict(eos_token=u.EOS, bos_token=u.BOS,
                          force_unk=True, sequential=True)
        self.seq_d.fit(self.corpus)
        self.tag1_d = Dict(eos_token=u.EOS, bos_token=u.BOS,
                           force_unk=True, sequential=True)
        self.tag1_d.fit(self.tag1_corpus)
        self.tag2_d = Dict(eos_token=u.EOS, bos_token=u.BOS,
                           force_unk=True, sequential=True)
        self.tag2_d.fit(self.tag2_corpus)
        # props
        self.batch_size = 10
        self.bptt = 5
        # datasets
        self.simple_dataset = BlockDataset(
            self.corpus, self.seq_d, self.batch_size, self.bptt)
        words, tags1, tags2 = [], [], []
        for s in self.tagged_corpus:
            words.append([tup[0] for tup in s])
            tags1.append([tup[1] for tup in s])
            tags2.append([tup[2] for tup in s])
        self.multi_dataset = BlockDataset(
            (words, tags1, tags2), (self.seq_d, self.tag1_d, self.tag2_d),
            self.batch_size, self.bptt)

    def test_target(self):
        for src, trg in self.simple_dataset:
            self.assertEqual(
                src[1:].data.sum(), trg[:-1].data.sum(),
                "Target batch is shifted by 1")

    def _recover_sentences(self, words):
        "Remove <bos><eos> markup and split into sequences"
        corpus, sent = [], []
        for word in words:
            if word == u.UNK:
                raise ValueError("Got UNK")
            if word == u.BOS:
                continue
            elif word == u.EOS:
                corpus.append(sent)
                sent = []
            else:
                sent.append(word)
        return corpus

    def _test_possibly_cropped_corpus(self, sents, msg):
        "compare recovered transformed dataset with original dataset"
        for idx, (sent1, sent2) in enumerate(zip(sents, self.corpus)):
            if idx == len(sents) - 1:
                # sent1 might be short if at end of the corpus
                self.assertListEqual(sent1, sent2[:len(sent1)], msg)
            else:
                self.assertListEqual(sent1, sent2, msg)

    def test_simple_data(self):
        indexed_words = self.simple_dataset.data.t().contiguous().view(-1)
        words = [self.seq_d.vocab[w] for w in indexed_words]
        self._test_possibly_cropped_corpus(
            self._recover_sentences(words),
            "Transformed data conforms to original data")

    def test_simple_mapping(self):
        batched = [[] for _ in range(self.batch_size)]
        for src, trg in self.simple_dataset:
            # bptt might differ for the last batch if not enough items
            self.assertEqual(
                src.size(1),    # source batch size
                self.batch_size,
                "Batch size conforms")
            by_col = [tuple(col.tolist()) for col in src.data.t()]
            for idx, seq in enumerate(by_col):
                seq = [self.seq_d.vocab[w] for w in seq]
                batched[idx] += seq
        # add last element from target
        for idx, b in enumerate(trg[-1].data.tolist()):
            batched[idx].append(self.seq_d.vocab[b])
        self._test_possibly_cropped_corpus(
            self._recover_sentences([w for b in batched for w in b]),
            "Batch-accessed ransformed data conforms to original data")

    def test_multi_mapping(self):
        ndims = len(self.multi_dataset.d)
        dimmed = [[[] for _ in range(self.batch_size)] for _ in range(ndims)]
        for src, trg in self.multi_dataset:
            self.assertEqual(len(src), ndims, "Source batches have same dims")
            self.assertEqual(len(trg), ndims, "Target batches have same dims")
            for dim, batch in enumerate(src):
                by_col = [tuple(col.tolist()) for col in batch.data.t()]
                for idx, seq in enumerate(by_col):
                    seq = [self.multi_dataset.d[dim].vocab[w] for w in seq]
                    dimmed[dim][idx] += seq
        for dim, trg_dim in enumerate(trg):
            for idx, b in enumerate(trg_dim[-1].data.tolist()):
                dimmed[dim][idx].append(self.multi_dataset.d[dim].vocab[b])
        # concatenate batches
        words = [[w for batch in dim for w in batch] for dim in dimmed]
        # remove reserved <bos><eos> tokens
        reserved = (u.EOS, u.BOS, u.UNK)
        words = [[w for w in dim if w not in reserved] for dim in words]
        # merge by tuples [(word, tag1, tag2, ...), ...]
        words = list(zip(*words))
        # flatten original data into tuples
        flattened = [tup for seq in self.tagged_corpus for tup in seq]
        self.assertEqual(
            words,
            flattened[:len(words)],
            "Batch-accessed transformed data conforms to flattened data")

    def test_splits(self):
        total = len(self.simple_dataset)
        train, test, valid = self.simple_dataset.splits(test=0.1, dev=0.1)
        places = len(self.corpus) * 0.1
        self.assertTrue(abs(int(total * 0.8) - len(train)) <= places, "train split")
        self.assertTrue(int(total * 0.1) - len(test) <= places, "test split")
        self.assertTrue(int(total * 0.1) - len(valid) <= places, "valid split")


class TestCompressionTable(unittest.TestCase):
    def setUp(self):
        # corpus
        self.corpus = [lorem.sentence().split() for _ in range(100)]
        self.nvals, self.batch_size = 3, 15
        self.tagged_corpus = \
            [[tuple([w, *self._encode_variables(self.nvals)]) for w in s]
             for s in self.corpus]
        self.conds = [conds for s in self.tagged_corpus for (w, *conds) in s]
        # compression table
        self.table = CompressionTable(self.nvals)
        self.hashed = [self.table.hash_vals(tuple(v)) for v in self.conds]

    def _encode_variables(self, nvals, card=3):
        import random
        return (random.randint(0, card) for _ in range(nvals))

    def test_mapping(self):
        for hashed, vals in zip(self.hashed, self.conds):
            self.assertEqual(tuple(vals), self.table.get_vals(hashed))

    def test_expand(self):
        # hashed conditions as tensor
        as_tensor = torch.tensor([h for h in self.hashed])
        # expand requires batched tensor
        num_batches, pad = divmod(len(as_tensor), self.batch_size)
        if pad != 0:            # pad tensor in case uneven length
            num_batches += 1
        # create 0-pad tensor and copy from source tensor
        t = torch.zeros([num_batches, self.batch_size]).long().view(-1)
        index = torch.tensor(list(range(len(as_tensor))))
        t.index_copy_(0, index, as_tensor)
        # expand
        conds = self.table.expand(t.view(-1, self.batch_size))
        # transform into original form for comparison
        conds = [c.view(-1) for c in conds]
        conds = [list(c) for c in zip(*conds)]
        self.assertEqual(self.conds, conds[:len(as_tensor)])


class TestPairedDataset(unittest.TestCase):
    def setUp(self):
        self.corpus = [lorem.sentence().split() for _ in range(100)]
        self.tagged_corpus = [[fake_tags(w) for w in s] for s in self.corpus]
        self.tag1_corpus = [[tup[1] for tup in s] for s in self.tagged_corpus]
        self.tag2_corpus = [[tup[1] for tup in s] for s in self.tagged_corpus]
        self.seq_d = Dict(eos_token=u.EOS, bos_token=u.BOS,
                          force_unk=True, sequential=True)
        self.seq_d.fit(self.corpus)
        self.tag1_d = Dict(eos_token=u.EOS, bos_token=u.BOS,
                           force_unk=True, sequential=True)
        self.tag1_d.fit(self.tag1_corpus)
        self.tag2_d = Dict(eos_token=u.EOS, bos_token=u.BOS,
                           force_unk=True, sequential=True)
        self.tag2_d.fit(self.tag2_corpus)

    def test_conditions(self):
        self.dataset = PairedDataset(
            self.corpus, (self.tag1_corpus, self.tag2_corpus),
            {'src': self.seq_d, 'trg': (self.tag1_d, self.tag2_d)})


class TestStratify(unittest.TestCase):
    def setUp(self):
        self.sents = []
        for _ in range(5000):
            sent = lorem.sentence().split()
            if sent not in self.sents:
                # avoid duplicates since `test_pairing` relies on sorting
                self.sents.append(sent)
        props = [0.1, 0.4, 0.3, 0.2]
        self.labels = np.random.multinomial(1, props, (len(self.sents))).argmax(1)
        d = Dict(pad_token='<PAD>').fit(self.sents)
        ld = Dict(sequential=False).fit(self.labels)
        self.dataset = PairedDataset(
            self.sents, self.labels, {'src': d, 'trg': ld}, batch_size=10)

    @staticmethod
    def reconstruct_set(dataset):
        sents = [[dataset.d['src'].vocab[w] for w in sent]
                 for sent in dataset.data['src']]
        labels = [dataset.d['trg'].vocab[l] for l in dataset.data['trg']]
        return sents, labels

    @staticmethod
    def argsort_dataset(sents, labels):
        sort = argsort([' '.join(sent) for sent in sents])
        return [sents[i] for i in sort], [labels[i] for i in sort]

    def test_pairing(self):
        self.dataset.sort_().shuffle_().stratify_()
        rec_sents, rec_labels = TestStratify.reconstruct_set(self.dataset)
        sort_rec_sents, sort_rec_labels = TestStratify.argsort_dataset(
            rec_sents, rec_labels)
        sort_sents, sort_labels = TestStratify.argsort_dataset(
            self.sents, self.labels)
        self.assertEqual(sort_rec_sents, sort_sents)
        self.assertEqual(sort_rec_labels, sort_labels)

    @staticmethod
    def dataset_mean_stddev(dataset):
        counts = defaultdict(list)
        for _, labels in dataset:
            for key, n in Counter(labels.data.tolist()).items():
                counts[key].append(n)

        from statistics import mean, stdev
        return {key: (mean(vals), stdev(vals)) for key, vals in counts.items()}

    @staticmethod
    def dataset_props(labels, batch_size):
        return {key: val / batch_size for key, val in Counter(labels).items()}

    def test_stratification(self):
        self.dataset.sort_().shuffle_().stratify_()
        mean_stddev = TestStratify.dataset_mean_stddev(self.dataset)
        props = TestStratify.dataset_props(self.dataset.data['trg'], len(self.dataset))
        for key in mean_stddev:
            # ignore stddev
            self.assertAlmostEqual(mean_stddev[key][0], props[key], delta=0.05)
