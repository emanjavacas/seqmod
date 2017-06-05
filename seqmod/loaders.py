
import os
import re
import json

from seqmod.misc.dataset import Dict, PairedDataset, default_sort_key


def identity(x):
    return x


def segmenter(tweet, level='char'):
    if level == 'char':
        return list(tweet)
    elif level == 'token':
        return tweet.split()
    else:
        raise ValueError


# Twisty
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


# Dataset
def load_dataset(src, trg, batch_size, max_size=100000, min_freq=5,
                 gpu=False, shuffle=True, sort_key=default_sort_key, **kwargs):
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
