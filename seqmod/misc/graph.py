
from graphviz import Digraph
import torch
from torch.autograd import Variable

from seqmod.misc.dataset import Dict


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot


def viz_encoder_decoder(**kwargs):
    from modules.encoder_decoder import EncoderDecoder
    num_layers, emb_dim, hid_dim, att_dim = 1, 12, 16, 16
    d = Dict(pad_token='<pad>').fit(['a'])
    m = EncoderDecoder(num_layers, emb_dim, hid_dim, att_dim, d, **kwargs)
    src, trg = torch.LongTensor([[0, 1]]), torch.LongTensor([[0, 1]])
    out = m(Variable(src), Variable(trg))
    return make_dot(out)


def viz_lm(**kwargs):
    from modules.lm import LM
    vocab, emb_dim, hid_dim = 10, 12, 16
    m = LM(vocab, emb_dim, hid_dim, **kwargs)
    out, _, _ = m(Variable(torch.LongTensor([[0, 1]])))
    return make_dot(out)


def viz_vae(**kwargs):
    from vae import SequenceVAE
    d = Dict(pad_token='<pad>').fit(['a'])
    num_layers, emb_dim, hid_dim, z_dim = 1, 12, 16, 16
    m = SequenceVAE(num_layers, emb_dim, hid_dim, z_dim, d)
    src = Variable(torch.LongTensor([[0, 1]]))
    logs, mu, logvar = m(src, src)
    z = m.encoder.reparametrize(mu, logvar)
    return make_dot(logs), make_dot(mu), make_dot(logvar), make_dot(z)
