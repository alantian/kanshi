#!/usr/bin/env python3

'''
抽取出句(=FLAGS.nb_ju 句詩)
'''

import glob
import os
import re
import sys

from tqdm import tqdm

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('input_file', '', 'input file')
gflags.DEFINE_string('output_file', '', 'output file')
gflags.DEFINE_integer('nb_ju', 2, 'number of ju. Can be either 2 or 4.')

FLAGS(sys.argv)

def is_valid_sent(sent):
    if len(sent) == 12 or len(sent) == 16:
        n = len(sent)
        if sent[n-1] == '。' and sent[n//2-1] == '，':
            return True
    return False

def chunk_poem(poem):
    result = []
    n = len(poem)
    last, now = -1, 0
    while now < n:
        if poem[now] == '。':
            result.append(poem[last+1:now+1])
            last = now
        now += 1

    result = [_ for _ in result if is_valid_sent(_)]  # each containing 2 ju
    gap = FLAGS.nb_ju // 2
    result = [
        ''.join(result[start: start + gap])
        for start in range(0, len(result), gap)
        if start + gap <= len(result)
    ]
    return result

def convert(poem_list):
    sent_list = []
    for poem in poem_list:
        sent_list.extend(chunk_poem(poem))
    return sent_list

def write(content, filepath):
    with open(filepath, 'w') as f:
        for line in content:
            print(line, file=f)

def main():

    poem_list = [line.strip() for line in open(FLAGS.input_file)]
    sent_list = convert(poem_list)

    write(sent_list, FLAGS.output_file)


if __name__ == '__main__':
    main()
