
import torch

import utils as u


def batchify(seqs, pad, align_right=False):
    max_length = max(len(x) for x in seqs)
    out = torch.LongTensor(len(seqs), max_length).fill_(pad)
    for i in range(len(seqs)):
        seq = torch.Tensor(seqs[i])
        seq_length = seq.size(0)
        offset = max_length - seq_length if align_right else 0
        out[i].narrow(0, offset, seq_length).copy_(seq)
    out = out.t().contiguous()
    return out


class Dataset(object):
    def __init__(self, src_data, tgt_data, batch_size, pad,
                 align_right=False, gpu=False):
        self.src = src_data
        assert len(self.src) == len(tgt_data), "src and tgt must be equal"
        self.tgt = tgt_data
        self.batch_size = batch_size
        self.pad = pad
        self.align_right = align_right
        self.gpu = gpu
        self.num_batches = len(self.src) // batch_size
        self.torched = False

    def _batchify(self, batch_data):
        if not self.torched:
            out = batchify(batch_data, self.pad, align_right=self.align_right)
        else:
            out = batch_data
        if self.gpu:
            out = out.cuda()
        return torch.autograd.Variable(out)

    def __getitem__(self, idx):
        assert idx < self.num_batches, "%d >= %d" % (idx, self.num_batches)
        batch_from = idx * self.batch_size
        batch_to = (idx+1)*self.batch_size
        src_batch = self._batchify(self.src[batch_from: batch_to])
        tgt_batch = self._batchify(self.tgt[batch_from: batch_to])
        return src_batch, tgt_batch

    def __len__(self):
        return self.num_batches

    @classmethod
    def from_disk(cls, path, batch_size, gpu=False):
        data = torch.load(path)
        src_data = data['src_data']
        tgt_data = data.get('tgt_data')
        instance = cls(src_data, tgt_data, batch_size, gpu=gpu)
        instance.torched = True
        return instance

    def to_disk(self, path):
        data = {'src_data': self.src_data, 'tgt_data': self.tgt_data}
        torch.save(data, path)


class DatasetMaker(object):
    def __init__(self, output_prefix, pad=u.PAD, eos=u.EOS, split=0.2):
        self.output_prefix = output_prefix
        self.pad = pad
        self.split = split
        self.sym2int = {}
        self.sym2int[u.PAD], self.sym2int[u.EOS] = 0, 1

    def encode_symbol(self, sym):
        if sym not in self.sym2int:
            code = len(self.sym2int)
            self.sym2int[sym] = code
        else:
            code = self.sym2int[sym]
        return code

    def parse_files(self, src_file, tgt_file):
        pass

    def torch(self, pairs, shuffle=True, sort=True):
        pass

    def save(self):
        # save dict
        # create dataset & save
        pass


if __name__ == '__main__':
    # load file parse it and store to file
    pass
