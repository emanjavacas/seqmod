
import os

import numpy as np

from gensim.models import Word2Vec
import fasttext


class Embedder(object):
    def __init__(self,
                 flavor='w2v', suffix='', directory='~/data/word_embeddings/'):
        if flavor not in ('w2v', 'ft'):
            raise NotImplementedError('Unsupported embedding option:', flavor)

        self.flavor = flavor
        self.fitted = False
        suffix = ('.' + suffix) if suffix else ''
        directory = os.path.expanduser(directory)
        self.path = os.path.join(
            directory, '{flavor}_model{suffix}'.format(
                flavor=flavor, suffix=suffix))

    def __getitem__(self, w):
        if not self.fitted:
            raise ValueError("Model hasn't been trained yet")
        try:
            return self.model[w]
        except KeyError:
            return self.default.copy()

    def load(self):
        if self.flavor == 'w2v':
            self.model = Word2Vec.load(self.path)
            self.model.init_sims(replace=True)
            self.size = self.model.size

        elif self.flavor == 'ft':
            self.model = fasttext.load_model(self.path + '.bin')
            self.size = self.model.dim

        self.fitted = True

    def fit(self, documents,
            alg='cbow', min_count=5, size=300, max_features=10000, window=5):

        assert alg in ('cbow', 'sg')

        if self.flavor == 'w2v':
            alg = 0 if alg == 'cbow' else 1
            self.model = Word2Vec(
                documents, min_count=min_count, size=size, window=window,
                max_vocab_size=max_features, sg=alg)
            self.model.save(self.path)
        elif self.flavor == 'ft':
            func = fasttext.cbow if alg == 'cbow' else fasttext.skipgram
            with open('/tmp/skiptrain.txt', 'w') as f:
                for d in documents:
                    f.write(' '.join(d) + '\n')
            self.model = func(
                input_file='/tmp/skiptrain.txt', output=self.path,
                min_count=min_count, dim=size, ws=window)

        self.size = size
        self.default = np.zeros(self.size, dtype='float64')
        self.fitted = True

        return self

    def transform(self, documents):
        X = []

        for d in documents:
            x = [self[w] for w in d]
            X.append(np.mean(x, axis=0))

        return np.array(X, dtype='float64')

    def fit_transform(self, documents, **kwargs):
        return self.fit(documents, **kwargs).transform(documents)


def load_embeddings(vocab, flavor, suffix, directory):
    """
    Load embeddings from a w2v model for model pretraining
    """
    size, embedder = 0, None

    if flavor == 'glove':
        embedder = {}
        directory = os.path.expanduser(directory)
        with open(os.path.join(directory, 'glove.%s.txt' % suffix), 'r') as f:
            for l in f:
                w, *vec = l.strip().split(' ')
                size = len(vec)
                embedder[w] = np.array(vec, dtype=np.float64)
    else:
        embedder = Embedder(
            flavor=flavor, suffix=suffix, directory=directory)
        embedder.load()
        size = embedder.size

    weight = np.zeros((len(vocab), size))
    for idx, w in enumerate(vocab):
        try:
            weight[idx] = embedder[w]
        except KeyError:
            pass                # default to zero vector
    return weight


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--flavor', default='w2v')
    parser.add_argument('--min_count', default=5, type=int)
    parser.add_argument('--size', default=200, type=int)
    parser.add_argument('--max_features', default=10000, type=int)
    parser.add_argument('--window', default=5, type=int)
    parser.add_argument('--alg', default='cbow', type=str)
    args = vars(parser.parse_args())

    print("Loading data...")
    from loaders import load_twisty
    tweets, _ = load_twisty()

    print("Creating model...")
    suffix = '{min_count}mc.{size}d.{window}w.{max_features}mf.{alg}'
    embedder = Embedder(flavor=args['flavor'], suffix=suffix.format(**args))

    print("Training...")
    del args['flavor']
    embedder.fit(tweets, **args)
