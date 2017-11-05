#!/usr/bin/env python3

import glob
import os
import re
import sys

from tqdm import tqdm

import gflags
FLAGS = gflags.FLAGS
gflags.DEFINE_string('input_files_pattern', '', 'glob pattern of input files')
gflags.DEFINE_string('output_file', '', 'output file')


def load_content(filepath):
    def line_is_start(line):
        return len(line) >= 3 and line[0] == '第' and line[-1] == '筆'

    def line_is_end(line):
        return line.startswith('[頁]卷,冊')

    def process(buf):
        # buf = [line if len(line) >= 5 else '' for line in buf]

        buf = [re.sub(r'\[\d*\]', '', line) for line in buf]

        buf = [re.sub(r'（[^）]*）', '', line) for line in buf]

        buf = [re.sub(r'〔[^〕]*〕', '', line) for line in buf]

        buf = [re.sub(r'　', '', line) for line in buf]

        sep = ':::'
        tot = sep.join(buf)
        pieces = tot.split(sep + sep)

        return [re.sub(sep, '', piece) for piece in pieces]

    result = []
    buf = []

    for raw_line in open(filepath):
        line = raw_line.rstrip()
        if line_is_start(line):
            buf = []
        elif line_is_end(line):
            result.extend(process(buf))
        else:
            if len(line) > 0 and line[0] in [' ', '　', '　']:
                pass
            else:
                buf.append(line)

    return result


def write(content, filepath):
    with open(filepath, 'w') as f:
        for line in content:
            print(line, file=f)


def main():
    FLAGS(sys.argv)

    content = []
    for filepath in tqdm(list(sorted(glob.glob(FLAGS.input_files_pattern)))):
        content.extend(load_content(filepath))

    write(content, FLAGS.output_file)


if __name__ == '__main__':
    main()
