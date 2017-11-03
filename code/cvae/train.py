#!/usr/bin/env python3

from collections import Counter
import os
from os import path
import random
import re
import sys

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
import nltk.translate.bleu_score as blue

from model import Decoder

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('vocab_file', '', 'input vocab file')
gflags.DEFINE_string('data_file', '', 'input data (h5) file')
gflags.DEFINE_string('model', '', 'model. can be rnnlm (or vae)')
gflags.DEFINE_integer('hidden_size', 300, 'size of hidden layer (and embedding)')
gflags.DEFINE_integer('latent_rep_size', 300, 'size of latent dim')
gflags.DEFINE_integer('n_layers', 4, 'number of layers')
gflags.DEFINE_integer('batch_size', 512, 'batch size')
gflags.DEFINE_integer('n_epoch', 100, 'number of epoch')
gflags.DEFINE_integer('log_interval', 10, 'log ever that many iters.')
gflags.DEFINE_boolean('show_sample', True, 'whether to show sampling at end of epoch')
gflags.DEFINE_integer('gpu_id', 0, 'id of gpu to use. -1 for cpu only.')
gflags.DEFINE_string('save_dir', './', 'dir to save model')
gflags.DEFINE_string('load_model', 'model_snapshot_iter_latest', 'model snapshot under save dir to load, if exits')
gflags.DEFINE_string('load_trainer', 'xxx_trainer_snapshot_iter_latest', 'trainer snapshot under save dir to load, if exits')


def gs(x, char_list):
    s = []
    for cid in x:
        if cid == 0:
            break
        s.append(char_list[int(cid)-1])
    s = ''.join(s)
    return s

'''
def find_nearest(y, data):
    y = [int(_) for _ in y]  # convert to list

    ny = []
    for _ in y:
        if _ == 0:
            break
        ny.append(str(int(_)))
    y = ny
    y_bigrams = set([tuple(y[i:i+2]) for i in range(len(y)-1)])

    ref_list = []
    ref_index_in_data = []
    for index, d in enumerate(data):
        nd = []
        for _ in d:
            if _ == 0:
                break
            nd.append(str(int(_)))
        nd_bigrams = set([tuple(nd[i:i+2]) for i in range(len(nd)-1)])
        has_overlap = len(y_bigrams & nd_bigrams) > 0
        if has_overlap:
            ref_list.append(nd)
            ref_index_in_data.append(index)

    if len(ref_list) == 0:
        return []

    scores = []
    for r in ref_list:
        scores.append(blue.sentence_bleu([r], y))
    scores = np.array(scores)

    index = np.argmax(scores)
    return data[ref_index_in_data[index]]
'''

def main():
    FLAGS(sys.argv)

    # 0. load dataset
    char_list = [line.strip().split('\t')[0] for line in open(FLAGS.vocab_file)]

    h5f = h5py.File(path.normpath(FLAGS.data_file), 'r')
    data = h5f['data'][:]
    train_data = data
    h5f.close()

    n, max_len = data.shape
    charset_size = len(char_list) + 1

    save_dir = path.normpath(FLAGS.save_dir)

    # 1. build model
    if FLAGS.model == 'rnnlm':
        model = Decoder(charset_size=charset_size, hidden_size=FLAGS.hidden_size,
        n_layers=FLAGS.n_layers)

    if FLAGS.gpu_id >= 0:
        chainer.cuda.get_device_from_id(FLAGS.gpu_id).use()
        model.to_gpu()

    load_model = path.join(save_dir, FLAGS.load_model)
    if os.path.exists(load_model):
        print('load model snapshot from %s' % load_model)
        serializers.load_npz(load_model, model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, FLAGS.batch_size)

    updater = training.StandardUpdater(
        train_iter, optimizer, device=FLAGS.gpu_id )
    trainer = training.Trainer(updater, stop_trigger=(FLAGS.n_epoch, 'epoch'), out=save_dir)

    trainer.extend(extensions.LogReport(
        trigger=(FLAGS.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/perp', 'elapsed_time']),
        trigger=(FLAGS.log_interval, 'iteration'))
    trainer.extend(extensions.snapshot(filename='trainer_snapshot_iter_{.updater.iteration}'))
    trainer.extend(extensions.snapshot(filename='trainer_snapshot_iter_latest'))
    trainer.extend(extensions.snapshot_object(target=model, filename='model_snapshot_iter_{.updater.iteration}'))
    trainer.extend(extensions.snapshot_object(target=model, filename='model_snapshot_iter_latest'))
    trainer.extend(extensions.ProgressBar())

    if FLAGS.show_sample:
        @chainer.training.make_extension()
        def sample(trainer):
            for temperature in [1.0, 1.3, 1.6, 1.9, 2.1]:
                print('sample (use random, t=%.2f):' % temperature)
                ys = model.sample(batch_size=2, use_random=True, temperature=temperature)
                for y in ys:
                    print('%s' % (gs(y, char_list)))
            print('sample (use max):')
            ys = model.sample(batch_size=1, use_random=False)
            for y in ys:
                print('%s' % (gs(y, char_list)))

        trainer.extend(
            sample, trigger=(1, 'epoch'))


    load_trainer = path.join(save_dir, FLAGS.load_trainer)
    if os.path.exists(load_trainer):
        print('load trainer snapshot from %s' % load_trainer)
        serializers.load_npz(load_trainer, trainer)

    print('start training')
    trainer.run()

import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
