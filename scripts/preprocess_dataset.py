
import os

import numpy as np

from seqmod.misc import Dict, text_processor
from seqmod.loaders import load_lines
import seqmod.utils as u


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('output')
    parser.add_argument('--max_size', default=1000000, type=int)
    parser.add_argument('--min_freq', default=1, type=int)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='char')
    args = parser.parse_args()

    processor = text_processor(
        lower=args.lower, num=args.num, level=args.level)
    d = Dict(max_size=args.max_size, min_freq=args.min_freq,
             eos_token=u.EOS, force_unk=True)

    trainpath = os.path.join(args.path, 'train.txt')
    testpath = os.path.join(args.path, 'test.txt')
    outputformat = (args.output + ".{}.npz").format

    if os.path.isfile(outputformat("train")):
        raise ValueError("Output train file already exists")
    if os.path.isfile(outputformat("test")):
        raise ValueError("Output test file already exists")

    print("Fitting dictionary")
    d.fit(load_lines(trainpath, processor=processor),
          load_lines(testpath, processor=processor))
    u.save_model(d, args.output + '.dict')

    print("Transforming train data")
    with open(outputformat("train"), 'wb+') as f:
        vector = []
        for line in d.transform(load_lines(trainpath, processor=processor)):
            vector.extend(line)
        np.save(f, np.array(vector))

    print("Transforming test data")
    with open(outputformat("test"), 'wb+') as f:
        vector = []
        for line in d.transform(load_lines(testpath, processor=processor)):
            vector.extend(line)
        np.save(f, np.array(vector))
