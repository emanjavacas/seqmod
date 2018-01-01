
import os
import re
import json
import time

from seqmod.misc import Dict, PairedDataset, text_processor
from seqmod import utils as u


class EmbeddingLoader(object):

    MODES = ('glove', 'fasttext')

    def __init__(self, fname, mode):
        if not os.path.isfile(fname):
            raise ValueError("Couldn't find file {}".format(fname))

        if mode.lower() not in EmbeddingLoader.MODES:
            raise ValueError("Unknown file mode {}".format(mode))

        self.fname = fname
        self.mode = mode.lower()

        self.has_header, self.use_model = False, False
        if self.mode == 'fasttext':
            self.has_header = True
            if fname.endswith('.bin'):
                self.use_model = True # for fasttext inference OOV

    def reader(self):
        with open(self.fname, 'r') as f:

            if self.has_header:
                next(f)

            for line in f:
                w, *vec = line.split(' ')

                yield w, vec

    def load_from_model(self, words, verbose=False):
        import numpy as np
        try:
            from fastText import load_model # defaul to official python bindings
        except ImportError:
            if verbose:
                print("Couldn't load official fasttext python bindings... "
                      "Defaulting to fasttext package")
            from fasttext import load_model

        start = time.time()
        model = load_model(self.fname)
        if verbose:
            print("Loaded model in {:.3f} secs".format(time.time()-start))

        vectors = np.array([model.get_word_vector(word) for word in words])
        return vectors, words

    def load(self, words=None, verbose=False):
        vectors, outwords, start = [], [], time.time()

        if words is not None:
            words = set(words)

            if verbose:
                print("Loading {} embeddings".format(len(words)))

        if words is not None and self.mode == 'fasttext' and self.use_model:
            vectors, outwords = self.load_from_model(words, verbose)

        else:
            for idx, (word, vec) in enumerate(self.reader()):
                if words is not None and word not in words:
                    continue

                try:
                    vec = list(map(float, vec))
                    vectors.append(vec)
                    outwords.append(word)
                except ValueError as e:
                    raise ValueError(str(e) + ' at {}:{}'.format(self.fname, idx))

        if verbose:
            print("Loaded {} embeddings in {:.3f}".format(
                len(outwords), time.time()-start))

        return vectors, outwords


def load_lines(path, max_len=None, processor=text_processor()):
    """Auxiliary function for sentence-per-line data"""
    lines = []

    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if processor is not None:
                line = processor(line)
            if not line or (max_len is not None and len(line) > max_len):
                continue
            lines.append(line)

    return lines


def load_split_data(path, batch_size, max_size, min_freq, max_len, gpu, processor):
    """
    Load corpus that is already splitted in 'train.txt', 'valid.txt', 'test.txt'
    """
    train_data = load_lines(
        os.path.join(path, 'train.txt'), max_len=max_len, processor=processor)
    valid_data = load_lines(
        os.path.join(path, 'valid.txt'), max_len=max_len, processor=processor)
    test_data = load_lines(
        os.path.join(path, 'test.txt'), max_len=max_len, processor=processor)

    d = Dict(pad_token=u.PAD, eos_token=u.EOS, bos_token=u.BOS,
             max_size=max_size, min_freq=min_freq, force_unk=True)
    d.fit(train_data, valid_data)

    train = PairedDataset(
        train_data, None, {'src': d}, batch_size, gpu=gpu)
    valid = PairedDataset(
        valid_data, None, {'src': d}, batch_size, gpu=gpu, evaluation=True)
    test = PairedDataset(
        test_data, None, {'src': d}, batch_size, gpu=gpu, evaluation=True)

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
