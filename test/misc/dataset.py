
import hashlib
from collections import Counter
import unittest

from seqmod.misc import dataset
from seqmod import utils as u


def fake_tags(w):
    return w, hashlib.md5(w.encode('utf-8')), hashlib.sha1(w.encode('utf-8'))


test_corpus = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "Curabitur scelerisque cursus lectus, ac efficitur felis congue vehicula.", "Etiam non fringilla mi.", "Curabitur blandit turpis id tellus pellentesque, nec pellentesque erat placerat.", "Duis fringilla mauris justo, ornare luctus lectus aliquam eget.", "Nullam egestas, velit eget hendrerit scelerisque, tellus nisl fringilla massa, sollicitudin ultricies lectus ante sed velit.", "Nullam malesuada hendrerit metus, vel auctor turpis tincidunt vel.", "Donec massa ipsum, fringilla a pharetra id, imperdiet nec metus.", "Aenean interdum nisi sed nunc congue tempor.", "Nullam nisi mi, vestibulum ac nunc ut, imperdiet interdum ligula.", "Pellentesque sed elementum neque.", "Pellentesque condimentum aliquet neque quis tincidunt.", "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Phasellus consectetur augue sit amet enim hendrerit, ac tincidunt nisl viverra.", "Ut at velit id mi pharetra tristique.", "Nam ut lacus sed orci consectetur venenatis.", "Aliquam viverra in enim sit amet pulvinar.",

    "Nunc a hendrerit velit.", "Suspendisse placerat condimentum sodales.", "Cras quis erat ac arcu eleifend ultrices vulputate quis ante.", "Nulla facilisi.", "Sed a vestibulum orci.", "Quisque porttitor leo vitae nisl tempor, in malesuada dolor volutpat.", "Nam vitae nisl dolor.", "Cras id rhoncus ante.", "Nulla hendrerit elementum maximus.", "Proin ac turpis sodales, suscipit erat nec, cursus justo.", "Donec finibus, ex eu consectetur ullamcorper, tortor ligula sollicitudin mauris, tincidunt molestie erat est sed tellus.", "Aenean tincidunt erat ut tempus sodales.",

    "Vivamus malesuada egestas mi in porttitor.", "Duis tincidunt neque vel feugiat finibus.", "Quisque eros erat, imperdiet eget sollicitudin id, cursus eu ex.", "Mauris sit amet velit at mi viverra tincidunt ac eget arcu.", "Suspendisse pulvinar molestie tempus.", "Duis eget lacinia massa.", "Aliquam sed risus scelerisque orci lacinia fermentum eget posuere nunc.", "Donec sed quam non mauris placerat pulvinar.",

    "Etiam vel molestie mauris.", "Quisque non posuere massa.", "Suspendisse cursus hendrerit dolor, vel molestie magna cursus at.", "Nulla maximus diam dolor.", "Nam id quam consectetur, consequat velit sed, varius orci.", "In accumsan nec ex id fringilla.", "Interdum et malesuada fames ac ante ipsum primis in faucibus.", "Donec a turpis rhoncus, sodales leo nec, sodales lacus.", "Nunc a odio quis ligula imperdiet maximus.", "Nulla scelerisque convallis quam ut ultrices.", "Fusce vel leo eget mi faucibus."
]

test_corpus = [s.split() for s in test_corpus]
tagged_test_corpus = [[fake_tags(w) for w in s] for s in test_corpus]


class TestDict(unittest.TestCase):
    def setUp(self):
        self.seq_vocab = Counter(w for s in test_corpus for w in s)
        self.seq_d = dataset.Dict(eos_token=u.EOS, bos_token=u.BOS,
                                  force_unk=True, sequential=True)
        self.seq_d.fit(test_corpus)
        self.seq_transformed = list(self.seq_d.transform(test_corpus))

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
            test_corpus,
            # remove <bos>, <eos> from transformed corpus
            [[self.seq_d.vocab[w] for w in s[1:-1]]
             for s in self.seq_d.transform(test_corpus)],
            "Transformed corpus matches word by word")


class TestBlockDataset(unittest.TestCase):
    def setUp(self):
        # dicts
        self.seq_d = dataset.Dict(eos_token=u.EOS, bos_token=u.BOS,
                                  force_unk=True, sequential=True)
        self.seq_d.fit(test_corpus)
        self.tag1_d = dataset.Dict(eos_token=u.EOS, bos_token=u.BOS,
                                   force_unk=True, sequential=True)
        self.tag1_corpus = [[tup[1] for tup in s] for s in tagged_test_corpus]
        self.tag1_d.fit(self.tag1_corpus)
        self.tag2_d = dataset.Dict(eos_token=u.EOS, bos_token=u.BOS,
                                   force_unk=True, sequential=True)
        self.tag2_corpus = [[tup[2] for tup in s] for s in tagged_test_corpus]
        self.tag2_d.fit(self.tag2_corpus)
        # props
        self.batch_size = 10
        self.bptt = 5
        # datasets
        self.simple_dataset = dataset.BlockDataset(
            test_corpus, self.seq_d, self.batch_size, self.bptt)
        words, tags1, tags2 = [], [], []
        for s in tagged_test_corpus:
            words.append([tup[0] for tup in s])
            tags1.append([tup[1] for tup in s])
            tags2.append([tup[2] for tup in s])
        self.multi_dataset = dataset.BlockDataset(
            (words, tags1, tags2), (self.seq_d, self.tag1_d, self.tag2_d),
            self.batch_size, self.bptt)

    def test_target(self):
        for src, trg in self.simple_dataset:
            self.assertEqual(src[1:], trg[:-1], "Target batch is shifted by 1")

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
        for idx, (sent1, sent2) in enumerate(zip(sents, test_corpus)):
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
        flattened = [tup for seq in tagged_test_corpus for tup in seq]
        self.assertEqual(
            words,
            flattened[:len(words)],
            "Batch-accessed transformed data conforms to flattened data")
