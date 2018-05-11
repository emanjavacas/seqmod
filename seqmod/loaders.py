
import os
import re
import json
import time

import numpy as np

from seqmod.misc.preprocess import text_processor
from seqmod.misc.dataset import Dict, PairedDataset
from seqmod import utils as u


class EmbeddingLoader(object):

    MODES = ('glove', 'fasttext', 'word2vec')

    def __init__(self, fname, mode=None):
        if not os.path.isfile(fname):
            raise ValueError("Couldn't find file {}".format(fname))

        basename = os.path.basename(fname).lower()

        if mode is None:
            if 'glove' in basename:
                mode = 'glove'
            elif 'fasttext' in basename or 'ft.' in basename:
                mode = 'fasttext'
            elif 'word2vec' in basename or 'w2v' in basename:
                mode = 'word2vec'
            else:
                raise ValueError("Unrecognized embedding type")

        if mode.lower() not in EmbeddingLoader.MODES:
            raise ValueError("Unknown file mode {}".format(mode))

        self.fname = fname
        self.mode = mode.lower()

        self.has_header = False
        if self.mode == 'fasttext':
            self.has_header = True

        self.use_model = False
        if fname.endswith('bin'):
            self.use_model = True  # for fasttext or w2v

    def reader(self):
        with open(self.fname, 'r') as f:

            if self.has_header:
                next(f)

            for line in f:
                w, *vec = line.split(' ')

                yield w, vec

    def load_from_model_ft(self, words, verbose=False):
        try:
            from fastText import load_model  # default to official python bindings
        except ImportError:
            if verbose:
                print("Couldn't load official fasttext python bindings... "
                      "Defaulting to fasttext package")
            try:
                from fasttext import load_model
            except ImportError:
                raise ValueError("No fasttext installation found. Please "
                                 "install one of `fastText`, `fasttext`")

        start = time.time()
        model = load_model(self.fname)
        if verbose:
            print("Loaded model in {:.3f} secs".format(time.time()-start))

        vectors = np.array([model.get_word_vector(word) for word in words])
        return vectors, words

    def load_from_model_w2v(self, words, maxwords=None, verbose=False):
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ValueError("No gensim installation found. Please install "
                             "`gensim` to load pretrained w2v embeddings.")
        start = time.time()
        model = KeyedVectors.load_word2vec_format(self.fname, binary=True)
        if verbose:
            print("Loaded model in {:.3f} secs".format(time.time()-start))

        if words is not None:
            vectors, outwords = [], []
            for word in words:
                try:
                    vectors.append(model[word])
                    outwords.append(word)
                except KeyError:
                    pass
        else:
            outwords = list(model.vocab.keys())
            if maxwords is not None:
                outwords = outwords[:min(maxwords, len(model.vocab)-1)]
            vectors = [model[w] for w in outwords]

        return np.array(vectors), outwords

    def load(self, words=None, maxwords=None, verbose=False):
        """
        Load embeddings.

        - words: list of str (optional)
        - maxwords: bool (optional), maximum number of words to be loaded. Only used if
            `words` isn't passed, otherwise all words in `words` will be (possibly) 
            loaded.
        """
        vectors, outwords, start = [], [], time.time()

        if words is not None:
            words = set(words)
            if verbose:
                print("Loading {} embeddings".format(len(words)))

        if words is not None and self.mode == 'fasttext' and self.use_model:
            vectors, outwords = self.load_from_model_ft(words, verbose)
        # always use gensim for word2vec (even if no restricted wordlist is provided)
        elif self.mode == 'word2vec' and self.use_model:
            vectors, outwords = self.load_from_model_w2v(
                words, maxwords=maxwords, verbose=verbose)
        else:
            for idx, (word, vec) in enumerate(self.reader()):
                if words is not None and word not in words:
                    continue
                if words is None and maxwords is not None and len(vectors) >= maxwords:
                    break
                try:
                    vec = list(map(float, vec))
                    vectors.append(vec)
                    outwords.append(word)
                except ValueError as e:
                    raise ValueError(str(e) + ' at {}:{}'.format(self.fname, idx))

        if verbose:
            print("Loaded {} embeddings in {:.3f} secs".format(
                len(outwords), time.time()-start))

        return vectors, outwords


def load_lines(path, processor=text_processor()):
    """Auxiliary function for sentence-per-line data"""
    if os.path.isdir(path):
        input_files = [os.path.join(path, f) for f in os.listdir(path)]
    elif os.path.isfile(path):
        input_files = [path]
    else:
        return

    for path in input_files:
        with open(os.path.expanduser(path)) as f:
            for line in f:
                line = line.strip()
                if processor is not None:
                    line = processor(line)
                if not line:
                    continue
                yield line


def load_split_data(path, batch_size, max_size, min_freq, max_len, device, processor):
    """
    Load corpus that is already splitted in 'train.txt', 'valid.txt', 'test.txt'
    """
    train = load_lines(os.path.join(path, 'train.txt'), max_len, processor)
    valid = load_lines(os.path.join(path, 'valid.txt'), max_len, processor)

    d = Dict(
        pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
        max_size=max_size, min_freq=min_freq, force_unk=True
    ).fit(train, valid)

    train = load_lines(os.path.join(path, 'train.txt'), max_len, processor)
    valid = load_lines(os.path.join(path, 'valid.txt'), max_len, processor)
    test = load_lines(os.path.join(path, 'test.txt'), max_len, processor)
    train = PairedDataset(train, None, {'src': d}, batch_size, device=device)
    valid = PairedDataset(valid, None, {'src': d}, batch_size, device=device)
    test = PairedDataset(test, None, {'src': d}, batch_size, device=device)

    return train.sort_(), valid.sort_(), test.sort_()


# Twisty
def load_twisty(path='/home/corpora/TwiSty/twisty-EN',
                min_len=0,
                concat=False,
                processor=lambda tweet: tweet,
                max_tweets=None):
    """
    Load twisty dataset with gender labels per tweet
    """
    src, trg, total_tweets = [], [], 0
    tweets_path = os.path.join(path, 'data/tweets/en/users_id/')
    tweet_fs = set(os.listdir(tweets_path))

    with open(os.path.join(path, 'TwiSty-EN.json'), 'r') as fp:
        metadata = json.load(fp)

    for user_id, user_metadata in metadata.items():
        if user_id + ".json" in tweet_fs:
            with open(os.path.join(tweets_path, user_id + '.json'), 'r') as fp:
                tweets = json.load(fp)['tweets']

            buf = []
            for tweet_id in user_metadata['confirmed_tweet_ids']:
                if max_tweets is not None and total_tweets >= max_tweets:
                    break

                buf.extend(processor(tweets[str(tweet_id)]['text']))
                total_tweets += 1

                if len(buf) > min_len:
                    src.append(buf), trg.append(user_metadata["gender"])
                    buf = []
                    continue
                if not concat:  # discard tweet
                    buf = []
                    continue

            else:               # when breaking in the inner loop
                continue
            break

    return src, trg


# Penn3
def _penn3_files_from_dir(path):
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            for f in os.listdir(os.path.join(path, d)):
                yield os.path.join(path, d, f)


def _penn3_lines_from_file(path):
    with open(path, 'r') as f:
        seq = []
        for idx, l in enumerate(f):
            line = l.strip()
            if not line or line.startswith('*'):
                continue
            elif line.startswith('='):
                if len(seq) != 0:
                    words, poses = zip(*seq)
                    yield words, poses
                    seq = []
            else:
                # remove brackets
                line = line.replace('[', '').replace(']', '')
                for t in line.split():
                    try:
                        word, pos = re.split(r'(?<!\\)/', t)
                        seq.append([word, pos])
                    except:
                        print("Couldn't parse line %d in file %s: %s" %
                              (idx, path, t))


def load_penn3(path, wsj=True, brown=True, swbd=False, atis=False):
    flags = [('wsj', wsj), ('brown', brown), ('swbd', swbd), ('atis', atis)]
    out, dirs = [], [d for d, flag in flags if flag]
    for d in dirs:
        out.extend(seq for f in _penn3_files_from_dir(os.path.join(path, d))
                   for seq in _penn3_lines_from_file(f))
    return out
