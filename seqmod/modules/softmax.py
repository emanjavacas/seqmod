
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from seqmod.modules.torch_utils import variational_dropout
from seqmod.modules.ff import Highway

from log_uniform import LogUniformSampler


class BaseSoftmax(nn.Module):
    """
    All Softmaxes must have the attributes:
       - `output_emb`, the actual output projection onto the vocabulary space
       - `tie_weights`, whether the model is setup to weight tying (might imply
            some extra projection)
       - `tied_weights`, whether the weights have already been tied
    """
    def forward(self, output, labels=False):
        raise NotImplementedError

    def tie_embedding_weights(self, embedding):
        """
        Actually tie the weights
        """
        if not self.tie_weights:
            raise ValueError("Module not setup for weight tying")

        if self.output_emb.weight.size() != embedding.weight.size():
            raise ValueError("Uncompatible weight sizes")

        self.tied_weights = True
        self.output_emb.weight = embedding.weight


class FullSoftmax(BaseSoftmax):
    """
    General output layer for Softmax-based models (LM, Decoder)
    It has options for adding a deepout layer previous to the softmax layers.
    """
    def __init__(self, hid_dim, emb_dim, vocab, tie_weights=False, dropout=0.0,
                 deepout_layers=0, deepout_act=None, maxouts=1):

        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.dropout = dropout
        self.tie_weights = tie_weights
        self.tied_weights = False  # flag to check if weights were tied before run

        super(FullSoftmax, self).__init__()

        if deepout_layers > 0:
            self.deepout = Highway(hid_dim, num_layers=deepout_layers,
                                   dropout=dropout, activation=deepout_act,
                                   # kwargs for custom MaxOut activation
                                   k=maxouts, in_dim=hid_dim, out_dim=hid_dim)

        if tie_weights:
            self.output_emb = nn.Linear(emb_dim, vocab)
            if emb_dim != hid_dim:
                # Insert a projection from hid_dim to emb_dim to have same dims
                # in both input and output embedding layers.
                logging.warn("When tying weights, output layer and "
                             "embedding layer should have equal size. "
                             "A projection layer will be inserted.")
                self.intermediate = nn.Linear(hid_dim, emb_dim)
        else:
            self.output_emb = nn.Linear(hid_dim, vocab)

    def forward(self, output, reshape=False, normalize=True):
        """
        output: Tensor((seq_len x) batch x hid_dim)
        reshape: whether to unflatten the seq_len and batch dims in the output
        normalize: whether to return log-probs (otherwise logits will be returned).
        """
        if self.tie_weights and not self.tied_weights:
            raise ValueError("Module should have tied weights")

        seq_len = 1 if output.dim() == 2 else output.size(0)
        output = output.view(-1, self.hid_dim)  # collapse seq_len and batch

        if hasattr(self, 'deepout'):
            output = self.deepout(output)
        if hasattr(self, 'intermediate'):
            output = self.intermediate(output)

        output = self.output_emb(output)  # ((seq_len *) batch x vocab)

        if normalize:
            output = F.log_softmax(output, dim=1)
        if reshape:
            output = output.view(seq_len, -1, self.vocab)

        return output


class MixtureSoftmax(BaseSoftmax):
    """
    Mixture of softmaxes to provide a high-rank approximatation to the actual output
    matrix that a Language Model is trying to factorize.

    Parameters:
    -----------
    - mixtures: int, number of softmaxes in the mixture
    """
    def __init__(self, hid_dim, emb_dim, vocab,
                 tie_weights=False, dropout=0.0, mixtures=5):
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.dropout = dropout
        self.mixtures = mixtures
        self.tie_weights = tie_weights
        self.tied_weights = False  # flag to check if weights were tied before run
        super(MixtureSoftmax, self).__init__()

        self.mixture_priors = nn.Linear(hid_dim, mixtures, bias=False)
        self.mixture_latent = nn.Linear(hid_dim, mixtures * emb_dim)
        self.output_emb = nn.Linear(emb_dim, vocab)

    def forward(self, output, normalize=True, reshape=False):
        if not normalize:
            raise ValueError("Mixture of Softmaxes cannot return logits")

        if self.tie_weights and not self.tied_weights:
            raise ValueError("Module should have tied weights")

        seq_len = 1 if output.dim() == 2 else output.size(0)
        output = output.view(-1, self.hid_dim)  # collapse seq_len and batch

        # Compute weights over mixtures: ((seq_len *) batch x mixture)
        priors = F.softmax(self.mixture_priors(output), dim=1)
        # Compute logits 1: (seq_len x batch x mixture * emb_dim)
        output = self.mixture_latent(output).view(
            seq_len, -1, self.mixtures * self.emb_dim)
        # Variational dropout
        output = variational_dropout(
            output, p=self.dropout, training=self.training)
        # Compute logits 2: ((seq_len *) batch * mixture x vocab)
        output = self.output_emb(output.view(-1, self.emb_dim))
        # Compute probabilities
        output = F.softmax(output, dim=1).view(-1, self.mixtures, self.vocab)
        # Mix: ((seq_len *) batch x vocab)
        output = (output * priors.unsqueeze(2).expand_as(output)).sum(1)
        # Transform probs to log-probs
        output = output.add_(1e-8).log()

        if reshape:
            # => (seq_len x batch x vocab)
            output = output.view(seq_len, -1, self.vocab)

        return output


# FIXME: Can't be put inside SampledSoftmax since it can't be pickled
_SAMPLER = None


class SampledSoftmax(FullSoftmax):
    def __init__(self, hid_dim, emb_dim, vocab, nsampled=8192, **kwargs):
        super(SampledSoftmax, self).__init__(hid_dim, emb_dim, vocab, **kwargs)

        global _SAMPLER
        _SAMPLER = LogUniformSampler(vocab)
        self.nsampled = nsampled

    def forward(self, output, targets=None, normalize=True, reshape=False):
        """
        Parameters:
        -----------
        output: ((seq_len *) batch_size x hid_dim)
        targets: ((seq_len *) batch_size)

        Returns:
        -------
        This module behaves exactly as FullSoftmax during evaluation but differently
        during training. During training, `targets` has to be passed and it's used
        to sample the new targets. The output is then the logits corresponding
        to the sampled targets and the sampled targets themselves, which are necessary
        to compute the loss.

        - output, new_targets: ((seq_len *) batch_size x nsampled), (nsampled)
        """
        seq_len = 1 if output.dim() == 2 else output.size(0)
        output = output.view(-1, self.hid_dim)  # collapse seq_len and batch

        if hasattr(self, 'deepout'):
            output = self.deepout(output)
        if hasattr(self, 'intermediate'):
            output = self.intermediate(output)

        if self.training:
            if targets is None:
                raise ValueError("SampledSoftmax requires `targets` during training")
            if reshape or normalize:
                raise ValueError("SampledSoftmax doesn't support `normalize` "
                                 "or `reshape` during training")
            return self.sampled(output, targets)

        output = self.output_emb(output)  # ((seq_len *) batch x vocab)

        if normalize:
            output = F.log_softmax(output, dim=1)
        if reshape:
            output = output.view(seq_len, -1, self.vocab)

        return output

    def sampled(self, output, targets, remove_accidental_match=True):
        """
        Parameters:
        -----------
        output: ((seq_len *) batch_size x hid_dim)
        targets: ((seq_len *) batch_size)

        Returns:
        --------
        logits: ((seq_len *) batch_size x nsampled + 1)  # adding the true class
        new_targets: ((seq_len *) batch_size)
        """
        # sample and wrap as variables
        sample_ids, true_freq, sample_freq = _SAMPLER.sample(
            self.nsampled, targets.data.cpu().numpy())
        sample_ids = Variable(output.data.new(sample_ids).long())
        true_freq = Variable(output.data.new(true_freq))
        sample_freq = Variable(output.data.new(sample_freq))

        # gather true labels and weights
        true_weights = self.output_emb.weight[targets, :]
        true_bias = self.output_emb.bias[targets]

        # gather sample labels and weights
        sample_weights = self.output_emb.weight[sample_ids, :]
        sample_bias = self.output_emb.bias[sample_ids]

        # calculate logits
        # row-wise dot-product of model output with output embedding
        # => diag(output @ true_weights.t()) + true_bias => (batch_size)
        true_logits = torch.sum(torch.mul(output, true_weights), dim=1) + true_bias
        # dot-product of model output with each of the sampled output embeddings
        # (batch_size x hid_dim) * (hid_dim x nsampled) => (batch_size x nsampled)
        sample_logits = torch.matmul(output, sample_weights.t()) + sample_bias

        # remove true targets from sample set
        if remove_accidental_match:
            acc_hits = _SAMPLER.accidental_match(
                targets.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            if len(acc_hits) > 0:
                sample_logits[list(zip(*acc_hits))] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_targets
        logits = torch.cat([true_logits.unsqueeze(1), sample_logits], dim=1)
        # zero tensor of size (batch_size), since true label is always first
        new_targets = Variable(output.data.new(output.size(0)).zero_().long())

        return logits, new_targets
