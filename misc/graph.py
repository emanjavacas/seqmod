
from graphviz import Digraph
import torch
from torch.autograd import Variable


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
    from encoder_decoder import EncoderDecoder
    num_layers, emb_dim, hid_dim, att_dim = 1, 12, 16, 16
    m = EncoderDecoder(num_layers, emb_dim, hid_dim, att_dim,
                       {'<pad>': 0, 'a': 1}, **kwargs)
    src, trg = torch.LongTensor([[0, 1]]), torch.LongTensor([[0, 1]])
    out = m(Variable(src), Variable(trg))
    dot = make_dot(out)
    return dot


def viz_lm(**kwargs):
    from lm import LM
    vocab, emb_dim, hid_dim = 10, 12, 16
    m = LM(vocab, emb_dim, hid_dim, **kwargs)
    out, _, _ = m(Variable(torch.LongTensor([[0, 1]])))
    dot = make_dot(out)
    return dot

