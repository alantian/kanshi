#!/usr/bin/env python3

import six

import numpy as np

import chainer
from chainer import variable
import chainer.cuda as cuda
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L
from chainer.cuda import cupy as cp

import numpy as np


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


EOS = 0


class Decoder(chainer.Chain):
    def __init__(self, charset_size, hidden_size, n_layers, dropout):
        super(Decoder, self).__init__()

        self.charset_size = charset_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        with self.init_scope():
            self.latent_rep_lin = L.Linear(n_layers * hidden_size)
            self.embedid = L.EmbedID(in_size=charset_size, out_size=hidden_size)
            self.gru = L.NStepGRU(n_layers=n_layers, in_size=hidden_size, out_size=hidden_size, dropout=dropout)
            self.W = L.Linear(hidden_size, charset_size)

    def __call__(self, ys):
        with chainer.cuda.get_device(ys[0]):
            batch_size = len(ys)
            eos = self.xp.array([EOS], 'i')
            ys_in = [F.concat([eos, y], axis=0) for y in ys]
            ys_out = [F.concat([y, eos], axis=0) for y in ys]
            eys = sequence_embed(self.embedid, ys_in)
            _, os = self.gru(None, eys)

            concat_os = F.concat(os, axis=0)
            concat_ys_out = F.concat(ys_out, axis=0)
            loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch_size

            chainer.report({'loss': loss.data}, self)
            n_words = concat_ys_out.shape[0]
            perp = self.xp.exp(loss.data * batch_size / n_words)
            chainer.report({'perp': perp}, self)
            return loss

    def webdnn_anchor(self):
        xid = variable.Variable(self.xp.full((1, 1), 0, 'i'))
        xs = self.embedid(xid)    # shape = (1, 1, hidden_size)

        hx = self.gru.init_hx(xs)
        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5] for w in self.gru]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5] for w in self.gru]
        hy, ys = self.gru.rnn(self.gru.n_layers, self.gru.dropout, hx, ws, bs, xs)
        y = ys[0]
        wy = self.W(y)    # shape = (1, charset_size)
        return [xid, hx], [wy, hy]

    def sample(self, batch_size=1, use_random=True, temperature=1.0, max_len=40, guide_ids=None, func_mask=None):
        guide_ids = [] if not guide_ids else guide_ids

        def normalize_p(p):
            return p / np.sum(p)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            ys = self.xp.full(batch_size, 0, 'i')
            result = []
            h = None
            for i in range(max_len):
                eys = self.embedid(ys)
                eys = F.split_axis(eys, batch_size, 0)
                h, ys = self.gru(h, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)

                if i < len(guide_ids):
                    ys = [guide_ids[i]] * batch_size
                    ys = self.xp.asarray(ys).astype('i')
                else:
                    if func_mask is None:
                        mask = self.xp.ones(wy.shape).astype('f')
                    else:
                        mask = np.ones(wy.shape).astype('f')
                        for row_id in range(batch_size):
                            mask[row_id] = func_mask([int(r[row_id]) for r in result])
                            if mask[row_id].sum() == 0:
                                mask[row_id, :] = 1.    # prevent error.
                        mask = self.xp.asarray(mask)
                    if use_random:
                        swy = cuda.to_cpu(F.softmax(wy.data, axis=1).data)
                        swy = np.power(swy, temperature)
                        swy = swy * cuda.to_cpu(mask)
                        ys = [np.random.choice(len(row), p=normalize_p(row)) for row in swy]
                        ys = self.xp.asarray(ys).astype('i')
                    else:
                        ys = self.xp.argmax(wy.data * self.xp.asarray(mask), axis=1).astype('i')
                result.append(ys)

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS tags
        outs = []
        for y in result:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def main():
    pass


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
