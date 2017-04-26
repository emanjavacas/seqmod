
import torch


class Beam(object):
    def __init__(self, width, bos, eos, gpu=False):
        self.width = width
        self.bos = bos
        self.eos = eos
        self.active = True
        self.scores = torch.FloatTensor(width).zero_()
        if gpu:
            init_state = torch.LongTensor(width).fill_(bos).cuda()
        else:
            init_state = torch.LongTensor(width).fill_(bos)
        self.beam_values = [init_state]  # output values at each beam
        self.source_beams = []  # backpointer to previous beam

    def _get_beam_at(self, step=-1):
        """
        Get beam at step `step`, defaulting to current step (= -1).
        """
        decoded = len(self.beam_values)
        assert step < decoded, "Only [%s] decoded steps" % decoded
        return self.beam_values[step]

    def __len__(self):
        """
        number of steps already decoded
        """
        return len(self.source_beams)

    def _new_beam(self, outs):
        """
        Computes a new beam based on the current model output and the hist.
        """
        width, vocab = outs.size()
        if len(self) > 0:
            # outs: (width x vocab) + scores: (width)
            beam_outs = outs + self.scores.unsqueeze(1).expand_as(outs)
        else:
            # all beams have same start values, just pick the 1st for perf
            beam_outs = outs[0]
        # compute best outputs over a flatten vector of size (width x vocab)
        # i.e. regardless their source beam
        scores, flatten_ids = beam_outs.view(-1).topk(self.width, dim=0)
        # compute source beam and best candidates for the next beam
        source_beams = flatten_ids / vocab
        beam_first_ids = source_beams * vocab
        beam = flatten_ids - beam_first_ids
        return scores, source_beams, beam

    def get_source_beam(self, step=-1):
        """
        Get entries in previous beam leading to a given step.
        """
        decoded = len(self.source_beams)
        assert step < decoded, "Only [%d] decoded steps" % decoded
        return self.source_beams[step]

    def get_current_state(self):
        """
        See _get_beam_at
        """
        return self._get_beam_at(step=-1)

    def finished(self, beam):
        """
        Finished criterion based on whether the last best hypothesis is EOS
        """
        return beam[0] == self.eos

    def advance(self, outs):
        """
        Runs a decoder step accumulating the path and the ids.
        """
        scores, source_beams, beam = self._new_beam(outs)
        if self.finished(beam):
            self.active = False
        self.scores = scores
        self.source_beams.append(source_beams)
        self.beam_values.append(beam)

    def get_hypothesis(self, idx):
        """
        Get hypothesis for `idx` entry in the current beam step.
        Note that the beam isn't mantained in sorted order.
        """
        assert idx <= self.width, "Beam has capacity [%d]" % self.width
        hypothesis = []
        for step in range(len(self) - 1, -1, -1):
            hypothesis.append(self._get_beam_at(step=step+1)[idx])
            idx = self.get_source_beam(step=step)[idx]
        return hypothesis[::-1]

    def decode(self, n=1):
        """
        Get n best hypothesis at current step.
        """
        assert n <= self.width, "Beam has capacity [%d]" % self.width
        scores, beam_ids = torch.sort(self.scores, dim=0, descending=True)
        best_scores, best_beam_ids = scores[:n], beam_ids[:n]
        return best_scores, [self.get_hypothesis(b) for b in best_beam_ids]
