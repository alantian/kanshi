#!/usr/bin/env python3

from collections import defaultdict
import os
from os import path

ytenx_sync_dir = path.join(path.dirname(path.realpath(__file__)), '../../data/ytenx-sync/')
misc_dir = path.join(path.dirname(path.realpath(__file__)), '../../data/misc')

zi_file = path.join(ytenx_sync_dir, 'kyonh/Dzih.txt')
xiaoyun_file = path.join(ytenx_sync_dir, 'kyonh/SieuxYonh.txt')
yunmu_file = path.join(ytenx_sync_dir, 'kyonh/YonhMiuk.txt')
guangyun_pingshui_file = path.join(misc_dir, 'guangyun-pingshuiyun.csv')
yiti_file = path.join(ytenx_sync_dir, 'jihthex/JihThex.csv')


def load_zi_to_xiaoyun_id_list():
    result = defaultdict(list)
    for line in open(zi_file):
        zi, xy_id, _, mean = line.strip().split(' ')[:4]
        xy_id = int(xy_id)
        result[zi].append((xy_id, mean))

    return result


def load_xiaoyun_id_to_yunmu():
    # yunmu = 韻目, not 韻母!
    result = dict()
    for line in open(xiaoyun_file):
        tks = line.strip().split(' ')
        xy_id = int(tks[0])
        yunmu = tks[4]
        result[xy_id] = yunmu
    return result


def load_yunmu_to_diao():
    # 韻目 -> 調 (1..4)
    result = dict()
    for line in open(yunmu_file):
        if not line.strip().startswith('#'):
            tks = line.strip().split(' ')
            yunmu = tks[0]
            diao = int(tks[2])
            result[yunmu] = diao
            if not yunmu.strip():
                print(line)
    return result


def load_yunmu_to_pingshui_yunmu(yunmu_list):
    mapping = {}
    for line_index, line in enumerate(open(guangyun_pingshui_file)):
        tks = line.strip().split(',')
        n = len(tks)
        assert n % 2 == 0
        for i in range(0, n, 2):
            pingshui_yunmu = tks[i]
            guangyun_yunmu = tks[i + 1]
            if pingshui_yunmu and guangyun_yunmu:
                mapping[guangyun_yunmu] = pingshui_yunmu

    result = {}
    for yunmu in yunmu_list:
        yunmu_part = yunmu[:1]
        assert yunmu_part in mapping
        result[yunmu] = mapping[yunmu_part]

    return result


def load_yiti_to_zhengti_list():
    result = defaultdict(list)
    for line in open(yiti_file):
        if not line.strip().startswith('#'):
            tks = line.strip().split(',')
            zi = tks[0]
            quandeng = list(tks[1])
            jiaodie = list(tks[2])
            jianti = list(tks[3])
            zhengti = list(tks[4])
            if zi:
                for _ in quandeng + jiaodie + jianti + zhengti:
                    result[zi].append(_)

    # manual fix some cases where morden usage is not recorded in rythm books
    for (a, b) in [('已', '巳'), ('笑', '𥬇'), ('疏', '䟽'), ('魂', '䰟'), ('島', '倒'), ('候', '𠋫'), ('瀟', '潚'), ('總', '緫'),
                   ('皓', '晧'), ('飆', '飊')]:
        result[a].append(b)
    return result


class ZiFeature(object):
    def __init__(self):
        self.zi_to_xiaoyun_id_list = load_zi_to_xiaoyun_id_list()
        self.xiaoyun_id_to_yunmu = load_xiaoyun_id_to_yunmu()
        self.yunmu_to_diao = yunmu_to_diao = load_yunmu_to_diao()
        self.yunmu_to_pingshui_yunmu = load_yunmu_to_pingshui_yunmu(yunmu_to_diao.keys())
        self.yiti_to_zhengti_list = load_yiti_to_zhengti_list()

        self.cache = {}

    def get_feature(self, input_):
        if isinstance(input_, list):
            return [self.get_feature(_) for _ in input_]
        else:
            if input_ in self.cache:
                return self.cache[input_]

            input_zi = input_
            result = []
            for zi in [input_zi] + self.yiti_to_zhengti_list.get(input_zi, []):
                for xiaoyun_id, mean in self.zi_to_xiaoyun_id_list.get(zi, []):
                    yunmu = self.xiaoyun_id_to_yunmu[xiaoyun_id]
                    diao = self.yunmu_to_diao[yunmu]
                    pingshui_yunmu = self.yunmu_to_pingshui_yunmu[yunmu]
                    result.append((diao, zi, mean, yunmu, pingshui_yunmu))

                if zi == input_zi and len(result) > 0:
                    break   # 非異體字，至此可矣。

            result = list(set(result))
            result = list(sorted(result, key=lambda _: len(_[1]), reverse=True))
            # sort, so char with longer mean comes first (I assume this means this is the more frequent usage)

            result = result[:1]    # only first
            self.cache[input_] = result
            return result


import numpy as np


def calc_mask_5(zf, prefix, char_list, char2id, offset=1):
    def pingze(zi1, zi2, t):
        assert t in ('dui', 'nian')
        f1 = zf.get_feature(zi1)
        f2 = zf.get_feature(zi2)
        f1_ping = any([_[0] == 1 for _ in f1])
        f1_ze = not f1_ping
        # f1_ze = any([_[0] != 1 for _ in f1])
        f2_ping = any([_[0] == 1 for _ in f2])
        f2_ze = not f2_ping
        # f2_ze = any([_[0] != 1 for _ in f2])

        if t == 'dui':
            result = f1_ping and f2_ze or f1_ze and f2_ping
        if t == 'nian':
            result = f1_ping and f2_ping or f1_ze and f2_ze

        # print('pingze %s %s %s -> %s' % (zi1, zi2, t, result))
        return result

    def ze(zi1):
        f1 = zf.get_feature(zi1)
        f1_ping = any([_[0] == 1 for _ in f1])
        f1_ze = not f1_ping
        # f1_ze = any([_[0] != 1 for _ in f1])
        return f1_ze

    def yayun(zi1, zi2):
        f1 = zf.get_feature(zi1)
        f2 = zf.get_feature(zi2)
        return any([_1[-1] == _2[-1] for _1 in f1 for _2 in f2])    # -2: 廣韻  -1:平水韻

    s = [char_list[int(_) - offset] for _ in prefix]
    index = len(prefix)
    ju_id = index // 6
    ju_pos = index % 6

    result = np.zeros((offset + len(char_list), ), dtype=np.float32)

    if ju_id >= 24:    # overlength
        result[:offset] = 1.
        return result

    if ju_pos == 5:
        if ju_id % 2 == 0:
            result[1 + char2id['，']] = 1.
        else:
            result[1 + char2id['。']] = 1.
        return result

    # pingze, yayun, etc
    if ju_id == 0:
        if ju_pos == 2:
            if ze(s[(ju_id) * 6 + 0]) and ze(s[(ju_id) * 6 + 1]):
                for cid in range(len(char_list)):
                    if not ze(char_list[cid]):
                        result[1 + cid] = 1.
            else:
                result[:] = 1.
        if ju_pos == 3:
            for cid in range(len(char_list)):
                if pingze(s[ju_id * 6 + 1], char_list[cid], 'dui'):
                    result[1 + cid] = 1.
                    # print('xxx: %s <-> %s dui' % (s[ju_id * 6 + 1], char_list[cid]))
            # print('yyy')
        else:
            result[:] = 1.
    else:
        if ju_pos == 2:
            if ze(s[(ju_id) * 6 + 0]) and ze(s[(ju_id) * 6 + 1]):
                for cid in range(len(char_list)):
                    if not ze(char_list[cid]):
                        result[1 + cid] = 1.
            else:
                result[:] = 1.
        if ju_pos == 1 or ju_pos == 3:
            for cid in range(len(char_list)):
                if pingze(s[(ju_id - 1) * 6 + ju_pos], char_list[cid], 'dui' if ju_id % 2 == 1 else 'nian'):
                    result[1 + cid] = 1.
        elif ju_pos == 4:
            if ju_id % 2 == 1 and ju_id > 1:
                for cid in range(len(char_list)):
                    if yayun(s[(ju_id - 2) * 6 + ju_pos], char_list[cid]):
                        result[1 + cid] = 1.
            elif ju_id == 1:
                for cid in range(len(char_list)):
                    if not ze(char_list[cid]):
                        result[1 + cid] = 1.
            else:
                for cid in range(len(char_list)):
                    if ze(char_list[cid]):
                        result[1 + cid] = 1.
        else:
            result[:] = 1.

    # kill repeating char.
    for p in prefix:
        if p != 1 + char2id['，'] and p != 1 + char2id['。']:
            result[p] = 0.
    return result

    assert False


def main():
    zi_feature = ZiFeature()
    cs = list('苟利國家生死以，豈因禍福避趨之。')
    print(zi_feature.get_feature(cs))


import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
