
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


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

    def custom_init(self):
        torch.nn.init.xavier_uniform(self.projection.weight)
        torch.nn.init.constant(self.projection.bias, 0)

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


