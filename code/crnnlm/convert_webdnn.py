#!/usr/bin/env python3

from collections import Counter
import os
from os import path
import random
import re
import sys

import chainer
from webdnn.frontend.chainer import ChainerConverter
from webdnn.backend import generate_descriptor

from model import Decoder

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('vocab_file', '', 'input vocab file')
gflags.DEFINE_string('model', '', 'model. can be rnnlm (or vae)')
gflags.DEFINE_integer('hidden_size', 300, 'size of hidden layer (and embedding)')
gflags.DEFINE_integer('n_layers', 4, 'number of layers')
gflags.DEFINE_float('dropout', 0.1, 'value for dropout')
gflags.DEFINE_string('save_dir', './', 'dir to save model')


def main():
    FLAGS(sys.argv)

    # 0. load dataset
    char_list = [line.strip().split('\t')[0] for line in open(FLAGS.vocab_file)]
    charset_size = len(char_list) + 1

    # 1. build model
    assert FLAGS.model in ('rnnlm')

    if FLAGS.model == 'rnnlm':
        model = Decoder(
            charset_size=charset_size, hidden_size=FLAGS.hidden_size, n_layers=FLAGS.n_layers, dropout=FLAGS.dropout
        )

    ins, outs = model.webdnn_anchor()
    graph = ChainerConverter().convert(ins, outs)
    exec_info = generate_descriptor("webgpu", graph)
    exec_info.save("./output")


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
