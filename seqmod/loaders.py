
import os
import re


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
