#!/usr/bin/env python3

#!/usr/bin/env python3

from collections import Counter
import json
import os
from os import path
import queue
import random
import re
import sys
import threading

from bottle import route, run, template, response
import chainer
from chainer import dataset, iterators, optimizers, serializers, training
from chainer.training import extensions
import chainer.cuda as cuda
from chainer.cuda import cupy as cp
import chainer.functions as F
import chainer.links as L
import h5py
import numpy as np
from tqdm import tqdm

from model import Decoder

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('vocab_file', '', 'input vocab file')
gflags.DEFINE_string('model', '', 'model. can be rnnlm (or vae)')
gflags.DEFINE_integer('hidden_size', 300, 'size of hidden layer (and embedding)')
gflags.DEFINE_integer('latent_rep_size', 300, 'size of latent dim')
gflags.DEFINE_integer('n_layers', 4, 'number of layers')
gflags.DEFINE_float('dropout', 0.1, 'value for dropout')
gflags.DEFINE_integer('batch_size', 512, 'batch size')
gflags.DEFINE_string('save_dir', './', 'dir to save model')
gflags.DEFINE_string('load_model', 'model_snapshot_iter_latest', 'model snapshot under save dir to load, if exits')
gflags.DEFINE_float('sample_t', 2.8, 't for sampling')
gflags.DEFINE_string('host', 'localhost', 'host to listen for server')
gflags.DEFINE_integer('port', 8000, 'port for server')
gflags.DEFINE_string('api_prefix', '/api/kanshi', 'prefix for handeling http api')
gflags.DEFINE_integer('buffer_size', 100, 'size of buffer')
gflags.DEFINE_integer('fill_size', 10, 'size of each single filling')
FLAGS(sys.argv)


def gs(x, char_list):
    s = []
    for cid in x:
        if cid == 0:
            break
        s.append(char_list[int(cid) - 1])
    s = ''.join(s)
    return s


def prepare_ctx():
    # 0. load dataset
    char_list = [line.strip().split('\t')[0] for line in open(FLAGS.vocab_file)]
    char_to_id = char2id = {c: i for i, c in enumerate(char_list)}
    charset_size = len(char_list) + 1

    # 1. build model
    if FLAGS.model == 'rnnlm':
        model = Decoder(
            charset_size=charset_size, hidden_size=FLAGS.hidden_size, n_layers=FLAGS.n_layers, dropout=FLAGS.dropout
        )

    save_dir = path.normpath(FLAGS.save_dir)
    load_model = path.join(save_dir, FLAGS.load_model)
    print('load model snapshot from %s' % load_model)
    serializers.load_npz(load_model, model)

    from lv import ZiFeature, calc_mask_5
    zf = ZiFeature()

    def func_mask(ys):
        return calc_mask_5(zf=zf, prefix=ys, char_list=char_list, char2id=char2id, offset=1)

    ctx = {'char_list': char_list, 'func_mask': func_mask, 'zf': zf, 'model': model}

    return ctx


ctx = prepare_ctx()
q = queue.Queue(maxsize=FLAGS.buffer_size)


def json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


def sample_many(nb=1):
    global ctx
    ys = ctx['model'].sample(
        batch_size=nb,
        use_random=True,
        temperature=FLAGS.sample_t,
        max_len=40,
        func_mask=ctx['func_mask'],
    )
    s = [gs(y, ctx['char_list']) for y in ys]
    return s


def fill_queue_func():
    while True:
        for s in sample_many(nb=FLAGS.fill_size):
            q.put(s)


@route(FLAGS.api_prefix + '/sample')
def sample():
    response.set_header('Access-Control-Allow-Origin', '*')
    poem = q.get()
    qsize = q.qsize()
    data = {'poem': poem, 'success': True, 'qsize': qsize}
    return json_dumps(data)


def main():
    threading.Thread(target=fill_queue_func).start()

    run(host=FLAGS.host, port=FLAGS.port)


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)