
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from seqmod.modules.torch_utils import variational_dropout


class MLP(nn.Module):
    """
    Standard MLP
    """
    def __init__(self, inp_size, hid_size, nb_classes,
                 nb_layers=1, dropout=0.0, act='relu'):
        self.inp_size, self.hid_size = inp_size, hid_size
        self.nb_layers, self.nb_classes = nb_layers, nb_classes
        self.dropout, self.act = dropout, act
        super(MLP, self).__init__()

        layers = []
        for i in range(nb_layers):
            layers.append(nn.Linear(inp_size, hid_size))
            inp_size = hid_size
        self.layers = nn.ModuleList(layers)

        self.output = nn.Linear(hid_size, nb_classes)

    def forward(self, inp):
        """
        :param inp: torch.FloatTensor (batch_size x inp_size)

        :return: torch.FloatTensor (batch_size x nb_classes)
        """
        # hidden layers
        for layer in self.layers:
            out = layer(inp)
            if self.act is not None:
                out = getattr(F, self.act)(out)
            if self.dropout > 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            inp = out

        # output projection
        out = self.output(out)

        return out


class MaxOut(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        """
        Implementation of MaxOut:
            h_i^{maxout} = max_{j \in [1, ..., k]} x^T W_{..., i, j} + b_{i, j}
        where W is in R^{D x M x K}, D is the input size, M is the output size
        and K is the number of pieces to max-pool from. (i.e. i ranges over M,
        j ranges over K and ... corresponds to the input dimension)

        Parameters:
        -----------
        in_dim: int, Input dimension
        out_dim: int, Output dimension
        k: int, number of "pools" to max over

        Returns:
        --------
        out: torch.Tensor (batch x k)
        """
        self.in_dim, self.out_dim, self.k = in_dim, out_dim, k
        super(MaxOut, self).__init__()
        self.projection = nn.Linear(in_dim, k * out_dim)

    def forward(self, inp):
        """
        Because of the linear projection we are bound to 1-d input
        (excluding batch-dim), therefore there is no need to generalize
        the implementation to n-dimensional input.
        """
        batch, in_dim = inp.size()
        # (batch x self.k * self.out_dim) -> (batch x self.out_dim x self.k)
        out = self.projection(inp).view(batch, self.out_dim, self.k)
        out, _ = out.max(2)
        return out


class Highway(torch.nn.Module):
    """
    Reference:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py

    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated
    combination of a linear transformation and a non-linear transformation
    of its input. y = g * x + (1 - g) * f(A(x)), where A
    is a linear transformation, `f` is an element-wise non-linearity,
    and `g` is an element-wise gate, computed as sigmoid(B(x)).

    Parameters
    ----------

    input_dim: int, The dimensionality of `x`.
    num_layers: int, optional, The number of highway layers.
    activation: str or class, if string it should be an activation function
        from torch.nn, otherwise it should be a class that will be instantiated
        with kwargs for each layer.
    dropout: float, dropout rate before the nonlinearity
    """
    def __init__(self, input_dim, num_layers=1, dropout=0.0, activation='ReLU',
                 **kwargs):
        self.input_dim = input_dim
        self.dropout = dropout
        super(Highway, self).__init__()

        layers = []
        for layer in range(num_layers):
            if isinstance(activation, type):  # custom activation class
                nonlinear = activation(**kwargs)
            else:               # assume string
                nonlinear = getattr(nn, activation)()

            linear = nn.Linear(input_dim, input_dim * 2)
            # We should bias the highway layer to just carry its input forward.
            # We do that by setting the bias on B(x) to be positive, because
            # that means `g` will be biased to be high, to we will carry the
            # input forward.  The bias on `B(x)` is the second half of the bias
            # vector in each Linear layer.
            linear.bias[input_dim:].data.fill_(1)
            linear.bias.custom = True

            layers.append(linear)
            layers.append(nonlinear)

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        current_input = inputs

        for i in range(0, len(self.layers), 2):
            layer, activation = self.layers[i], self.layers[i+1]
            proj, linear = layer(current_input), current_input
            proj = F.dropout(proj, p=self.dropout, training=self.training)
            nonlinear = activation(proj[:, 0:self.input_dim])
            gate = F.sigmoid(proj[:, self.input_dim:(2 * self.input_dim)])

            # apply gate
            current_input = gate * linear + (1 - gate) * nonlinear

        return current_input

# gracefully taken from:
# https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
class GradReverse(Function):
    "Implementation of GRL from DANN (Domain Adaptation Neural Network) paper"
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    """
    GRL must be placed between the feature extractor and the domain classifier
    """
    return GradReverse.apply(x)


class OutputSoftmax(nn.Module):
    """
    General output layer for Softmax-based models (LM, Decoder)
    It has options for adding a a deepout previous to the softmax layers.
    """
    def __init__(self, hid_dim, emb_dim, vocab, tie_weights=False, dropout=0.0,
                 mixture=0, deepout_layers=0, deepout_act=None, maxouts=1):

        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.tie_weights = tie_weights
        self.dropout = dropout
        self.mixture = mixture
        self.has_mixture = bool(mixture)
        self.tied_weights = False  # flag to check if weights were tied before run
        self.has_deepout = False
        self.has_intermediate_proj = False

        super(OutputSoftmax, self).__init__()

        if deepout_layers > 0:
            self.has_deepout = True
            self.deepout = Highway(hid_dim, num_layers=deepout_layers,
                                   dropout=dropout, activation=deepout_act,
                                   # kwargs for custom MaxOut activation
                                   k=maxouts, in_dim=hid_dim, out_dim=hid_dim)

        if self.has_mixture:
            self.mixture_priors = nn.Linear(hid_dim, mixture, bias=False)
            self.mixture_latent = nn.Linear(hid_dim, mixture * emb_dim)

        if tie_weights:
            self.output_emb = nn.Linear(emb_dim, vocab)
            if emb_dim != hid_dim and not self.has_mixture:
                # Insert a projection from hid_dim to emb_dim to have same dims
                # in both input and output embedding layers.
                # Mixture of softmaxes doesn't have this constraint
                # as it always projects hid_dim to emb_dim (* mixtures)
                logging.warn("When tying weights, output layer and "
                             "embedding layer should have equal size. "
                             "A projection layer will be insterted.")
                self.intermediate = nn.Linear(hid_dim, emb_dim)
                self.has_intermediate_proj = True
        else:
            # mixture softmax introduces a layer before output embeddings
            # projecting to the embedding dimension
            inp_dim = emb_dim if self.has_mixture else hid_dim
            self.output_emb = nn.Linear(inp_dim, vocab)

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

    def forward(self, output, reshape=False, normalize=True):
        """
        output: Tensor((seq_len x) batch x hid_dim)
        reshape: whether to unflatten the seq_len and batch dims in the output
        normalize: whether to return log-probs (otherwise logits will be returned).
            It cannot be False for mixture softmaxes (an error is thrown since it
            would probably imply a design error in client code).
        """
        if self.tie_weights and not self.tied_weights:
            raise ValueError("Module should have tied weights")

        seq_len = 1 if output.dim() == 2 else output.size(0)
        output = output.view(-1, self.hid_dim)  # collapse seq_len and batch

        if self.has_deepout:
            output = self.deepout(output)

        if self.has_intermediate_proj:
            output = self.intermediate(output)

        if self.has_mixture:
            if not normalize:
                raise ValueError("Mixture of Softmaxes cannot return logits")

            # Compute weights over mixtures: ((seq_len *) batch x mixture)
            priors = F.softmax(self.mixture_priors(output), dim=1)
            # Compute logits 1: (seq_len x batch x mixture * emb_dim)
            output = self.mixture_latent(output).view(
                seq_len, -1, self.mixture * self.emb_dim)
            # Variational dropout
            output = variational_dropout(
                output, p=self.dropout, training=self.training)
            # Compute logits 2: ((seq_len *) batch * mixture x vocab)
            output = self.output_emb(output.view(-1, self.emb_dim))
            # Compute probabilities
            output = F.softmax(output, dim=1).view(-1, self.mixture, self.vocab)
            # Mix: ((seq_len *) batch x vocab)
            output = (output * priors.unsqueeze(2).expand_as(output)).sum(1)
            # Transform log-probs to probs
            output = output.add_(1e-8).log()

        else:
            # ((seq_len *) batch x vocab)
            output = self.output_emb(output)

            if normalize:
                output = F.log_softmax(output, dim=1)

        if reshape:
            # => (seq_len x batch x vocab)
            output = output.view(seq_len, -1, self.vocab)

        return output
