
import torch
import torch.nn as nn


def word_dropout_mask(X, dropout_rate, reserved_codes=()):
    """
    Computes a binary mask across batch examples based on a
    bernoulli distribution with mean equal to dropout.
    """
    probs = X.new(*X.size()).zero_().float() + dropout_rate
    # zero reserved_codes (avoid dropping reserved symbols)
    probs[sum((X == x) for x in reserved_codes)] = 0
    return probs.bernoulli_().byte()


def word_dropout(inp, target_code, p=0.0, training=True,
                 reserved_codes=(), lengths=None):
    """
    Applies word dropout to an input Variable. Dropout isn't constant
    across batch examples. This is only to be used to drop input symbols
    (i.e. before the embedding layer)

    Parameters:
    -----------
    - inp: torch.LongTensor
    - target_code: int, code to use as replacement for dropped timesteps
    - dropout: float, dropout rate
    - reserved_codes: tuple of ints, ints in the input that should never
        be dropped
    - training: bool
    """
    if not training or p == 0:
        return inp

    mask = word_dropout_mask(
        inp.data, dropout_rate=p, reserved_codes=reserved_codes)

    return inp.masked_fill(torch.autograd.Variable(mask), target_code)


class Embedding(nn.Embedding):
    def __init__(self, *args, d, word_dropout=0.0, **kwargs):
        super(Embedding, self).__init__(
            *args, padding_idx=d.get_pad(), **kwargs)

        self.dictionary = d
        self.word_dropout = word_dropout
        self.target_code = self.dictionary.get_unk()
        codes = [self.dictionary.get_eos(),
                 self.dictionary.get_bos(),
                 self.dictionary.get_pad()]
        self.reserved_codes = tuple([c for c in codes if c is not None])

    def forward(self, inp):
        inp = word_dropout(
            inp, self.target_code, reserved_codes=self.reserved_codes,
            p=self.word_dropout, training=self.training)

        return super(Embedding, self).forward(inp)

    def load_embeddings(self, weight, words, verbose=False):
        """
        Load embeddings from a weight matrix with words `words` as rows.

        Parameters
        -----------
        - weight: (vocab x emb_dim)
        - words: list of word indices corresponding to each row in `weight`
        """
        # wrap in tensor
        if isinstance(weight, list):
            weight = torch.Tensor(weight).float()
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight).float()
        # check embedding size
        assert weight.size(1) == self.embbeding_size, \
            "Mismatched embedding dim {} for model with dim {}".format(
                (weight.size(1), self.embedding_size))

        self_idxs, other_idxs = [], []
        for other_idx, word in enumerate(words):
            try:
                self_idxs.append(self.dictionary.s2i[word])
                other_idxs.append(other_idx)
            except KeyError:
                if verbose:
                    print("Couldn't find {} in dictionary".format(word))
                pass

        other_idxs = torch.LongTensor(other_idxs)
        self_idxs = torch.LongTensor(self_idxs)
        self.weight.data[self_idxs] = weight[other_idxs]
