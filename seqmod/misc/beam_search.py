
import torch


class Beam(object):
    """
    Beam class for performing beam search

    Parameters
    -----------
    width: int, beam buffer size. The higher the better the chances are of
        actually decoding the best sequence but also the bigger the memory
        footprint
    prev: int, integer token to use as first decoding step
    eos: int or None, integer corresponding to the <eos> symbol in the
        vocabulary. It will be used as terminating criterion for the decoding
    """
    def __init__(self, width, prev, batch_size=1, eos=None, gpu=False):
        self.width = width
        self.batch_size = batch_size
        self.eos = eos
        self.active = True
        self.scores = torch.FloatTensor(width * batch_size).zero_()
        init_state = torch.LongTensor(width * batch_size).fill_(prev)
        if gpu:
            init_state = init_state.cuda()
        # output values at each beam
        self.beam_values = [init_state]
        # backpointer to previous beam
        self.source_beams = []
        self._finished = torch.LongTensor(batch_size).zero_()

    def __len__(self):
        """
        number of steps already decoded
        """
        return len(self.source_beams)

    def get_source_beam(self, step=-1):
        """
        Get entries in previous beam leading to a given step.
        """
        if step >= len(self.source_beams):
            raise ValueError(
                "Only {} decoded steps".format(len(self.source_beams)))

        return self.source_beams[step]

    def get_current_state(self):
        """
        Return current beam step
        """
        return self.beam_values[-1]

    def finished(self, beam):
        """
        Finished criterion based on whether the last best hypothesis is EOS

        beam: (batch_size x width)
        """
        self._finished += (beam[:, 0] == self.eos).long()
        return self._finished.nonzero().nelement() == self.batch_size

    def advance(self, outs):
        """
        Runs a decoder step accumulating the path and the ids.

        outs: (width * batch_size x vocab)
        """
        if len(self) == 0:
            # all beams have same start value, just pick the 1st
            beam_outs = outs.view(self.width, self.batch_size, -1)[0]
        else:
            # accumulate scores
            beam_outs = outs + self.scores.unsqueeze(1).expand_as(outs)
            # EOS nihilation (adapted from OpenNMT)
            for i in range(len(beam_outs)):
                if self.eos is not None and self.beam_values[-1][i] == self.eos:
                    beam_outs[i] = -1e20

            # -> (batch_size, (width *) vocab)
            beam_outs = beam_outs.view(self.width, self.batch_size, -1) \
                                 .transpose(0, 1).contiguous() \
                                 .view(self.batch_size, -1)

        vocab = outs.size(-1)
        # compute best outputs over a flatten vector of shape
        # (batch_size x (width *) vocab), i.e. regardless their source beam
        # scores, flattened_ids: (batch_size x width)
        scores, flattened_ids = beam_outs.topk(self.width, dim=1)

        # update scores
        self.scores = scores.t().contiguous().view(-1)

        # update source_beams (can't do // on tensors)
        offset = torch.arange(0, self.batch_size, out=outs.new()).long()
        offset = offset.unsqueeze(1).repeat(1, self.width).view(-1) * self.width
        source_beams = (flattened_ids / vocab).t().contiguous().view(-1) + offset
        self.source_beams.append(source_beams)

        # update current beam symbols
        beam_values = (flattened_ids % vocab)

        if self.finished(beam_values):
            self.active = False

        self.beam_values.append(beam_values.t().contiguous().view(-1))

    def get_hypothesis(self, idxs):
        """
        Get hypothesis for `idx` entry in the current beam step.
        Note that the beam isn't mantained in sorted order.

        idxs: (batch_size)
        """
        hypothesis = []
        for step in range(len(self)-1, -1, -1):
            hypothesis.append(self.beam_values[step+1][idxs].tolist())
            idxs = self.get_source_beam(step=step)[idxs]

        return hypothesis[::-1]

    def decode(self, n=1):
        """
        Get n best hypothesis at current step.
        """
        if n > self.width:
            raise ValueError("Beam has only capacity {}".format(self.width))

        # (batch_size x width)
        scores = self.scores.view(self.width, self.batch_size).t()
        scores, beam_ids = torch.sort(scores, dim=1, descending=True)
        best_scores = scores[:, :n].tolist()

        offset = torch.arange(0, self.batch_size, out=beam_ids.new()).long()
        offset = offset.unsqueeze(1).repeat(1, self.width).view(-1) * self.width
        beam_ids = beam_ids.view(-1) + offset
        # (width x batch_size)
        best_beam_ids = beam_ids.view(self.batch_size, self.width).t().contiguous()[:n]

        hyps = [self.get_hypothesis(b) for b in best_beam_ids]
        # (width x seq_len x batch) -> (batch x width x seq_len)
        hyps = torch.LongTensor(hyps).transpose(0, 1).transpose(0, 2).tolist()

        return best_scores, hyps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()

    from seqmod.utils import load_model, wrap_variables
    import random
    random.seed(1000)
    from random import choice, randint
    model = load_model(args.model)
    d = model.encoder.embeddings.d

    words = []
    for _ in range(4):
        words.append(''.join(choice(d.vocab) for _ in range(randint(5, 10))))
    words = list(d.transform(words))

    src, src_lengths = wrap_variables(
        d.pack(words, return_lengths=True), volatile=True)

    scores, hyps, _ = model.translate_beam(src, src_lengths)
    # beam = model.translate_beam(src, src_lengths)
