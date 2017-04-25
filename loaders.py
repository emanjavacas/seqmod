
import os
import json

import numpy as np

from misc.dataset import Dict, PairedDataset

from w2v import Embedder


def identity(x):
    return x


def segmenter(tweet, level='char'):
    if level == 'char':
        return list(tweet)
    elif level == 'token':
        return tweet.split()
    else:
        raise ValueError


# Data loaders
def load_twisty(path='/home/corpora/TwiSty/twisty-EN',
                min_len=0,
                level='token',
                concat=False,
                processor=identity):
    """
    Load twisty dataset with gender labels per tweet
    """
    tweets_path = os.path.join(path, 'data/tweets/en/users_id/')
    tweet_fs = set(os.listdir(tweets_path))
    with open(os.path.join(path, 'TwiSty-EN.json'), 'r') as fp:
        metadata = json.load(fp)
    src, trg = [], []
    for user_id, user_metadata in metadata.items():
        if user_id + ".json" in tweet_fs:
            with open(os.path.join(tweets_path, user_id + '.json'), 'r') as fp:
                tweets = json.load(fp)['tweets']
            buf = []
            for tweet_id in user_metadata['confirmed_tweet_ids']:
                tweet = processor(tweets[str(tweet_id)]['text'])
                tweet = segmenter(tweet, level=level)
                buf.extend(tweet)
                if len(buf) > min_len:
                    src.append(buf), trg.append([user_metadata["gender"]])
                    buf = []
                    continue
                if not concat:  # discard tweet
                    buf = []
                    continue

    return src, trg


def sort_key(pair):
    """
    Sort examples by tweet length
    """
    src, trg = pair
    return len(src)


def load_dataset(src, trg, batch_size, max_size=100000, min_freq=5,
                 gpu=False, shuffle=True, sort_key=sort_key, **kwargs):
    """
    Wrapper function for dataset with sensible, overwritable defaults
    """
    tweets_dict = Dict(pad_token='<pad>', eos_token='<eos>',
                       bos_token='<bos>', max_size=max_size, min_freq=min_freq)
    labels_dict = Dict(sequential=False, force_unk=False)
    tweets_dict.fit(src)
    labels_dict.fit(trg)
    d = {'src': tweets_dict, 'trg': labels_dict}
    splits = PairedDataset(src, trg, d, batch_size, gpu=gpu).splits(
        shuffle=shuffle, sort_key=sort_key, **kwargs)
    return splits


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
