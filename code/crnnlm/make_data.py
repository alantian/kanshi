#!/usr/bin/env python3

from collections import Counter
import os
import re
import sys

import numpy as np
import h5py

import lv

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('input_file', '', 'input file.')
gflags.DEFINE_string('vocab_file', '', 'output vocab file')
gflags.DEFINE_string('data_file', '', 'output data (h5) file')


def main():
    FLAGS(sys.argv)

    sent_list = [line.strip() for line in open(FLAGS.input_file)]

    char_counter = Counter([c for sent in sent_list for c in sent])
    char_count_list = (char_counter.most_common())
    char_count_list.sort(key=lambda elem: (elem[1], ord(elem[0])))
    char_list = []
    with open(FLAGS.vocab_file, 'w') as f:
        for char, count in char_count_list:
            print('%s\t%s' % (char, count), file=f)
            char_list.append(char)
    char2id = {c: i for i, c in enumerate(char_list)}

    max_len = max([len(sent) for sent in sent_list])
    n = len(sent_list)
    max_len = max_len + 1

    data = np.zeros((n, max_len), dtype=np.int32)    # content is 1 + char2id[c]. 0 is for padding
    for sent_id, sent in enumerate(sent_list):
        indices = []
        for c in sent:
            indices.append(1 + char2id[c])
        if len(indices) < max_len:
            indices.extend((max_len - len(indices)) * [0])
        data[sent_id] = indices

    data_file = FLAGS.data_file
    assert data_file.endswith('.h5')
    h5f = h5py.File(data_file, 'w')
    h5f.create_dataset('data', data=data)
    h5f.close()

    print('data size is', data.shape)
    print('char size is', len(char_list))


if __name__ == '__main__':
    main()
