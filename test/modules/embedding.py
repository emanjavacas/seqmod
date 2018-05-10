
import unittest

import torch
import lorem

from seqmod.modules import embedding
from seqmod.loaders import EmbeddingLoader
from seqmod import utils as u
from seqmod.misc.dataset import Dict, pad_sequential_batch


class EmbeddingTest(unittest.TestCase):
    def setUp(self):
        text = [lorem.sentence() for _ in range(100)]
        self.d = Dict(pad_token=u.PAD).fit(text)
        self.corpus = list(self.d.transform(text))
        self.num_words = sum(len(s.split()) for s in text)
        self.word_lengths = [len(s.split()) for s in text]
        self.max_word_len = max(len(w) for s in text for w in s.split())
        self.max_seq_words = max(len(s.split()) for s in text)


class TestEmbeddingFromDict(EmbeddingTest):
    def setUp(self):
        super(TestEmbeddingFromDict, self).setUp()
        self.filepath = 'test/data/glove.test1000.100d.txt'

    def test_loading_embeddings(self):
        emb = embedding.Embedding.from_dict(self.d, 100)
        emb.init_embeddings_from_file(self.filepath)

        weights, words = EmbeddingLoader(self.filepath, 'glove').load()
        weights = torch.Tensor(weights)
        by_word = dict(zip(words, weights))

        for weight, word in zip(emb.weight.data, emb.d.vocab):
            if word in by_word:
                self.assertTrue(torch.equal(weight, by_word[word]))


class TestComplexEmbedding(EmbeddingTest):
    def setUp(self):
        super(TestComplexEmbedding, self).setUp()
        inp, lengths = pad_sequential_batch(self.corpus, self.d.get_pad(), True, False)
        self.inp = torch.tensor(inp)
        self.lengths = torch.tensor(lengths)

    def _test_embedding_dimensions(self, emb):
        output, lengths = emb(self.inp, self.lengths)
        max_seq_words, batch, emb_dim = output.size()
        self.assertEqual(max_seq_words, self.max_seq_words)
        self.assertEqual(batch, len(self.corpus))
        self.assertEqual(emb_dim, emb.embedding_dim)
        self.assertEqual(lengths, self.word_lengths)

    def test_rnn_embeddings_dimensions_with_context(self):
        emb = embedding.RNNEmbedding.from_dict(self.d, 100, contextual=True)
        self._test_embedding_dimensions(emb)

    def test_rnn_embeddings_dimensions_without_context(self):
        emb = embedding.RNNEmbedding.from_dict(self.d, 100, contextual=False)
        self._test_embedding_dimensions(emb)

    def test_rnn_embeddings_dimensions_bidi(self):
        emb = embedding.RNNEmbedding.from_dict(self.d, 100, bidirectional=True)
        self._test_embedding_dimensions(emb)

    def test_cnn_embeddings(self):
        emb = embedding.CNNEmbedding.from_dict(self.d, 24)
        self._test_embedding_dimensions(emb)
