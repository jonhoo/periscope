#!/usr/bin/env python3

import argparse
import numpy
import re
import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--confusion', type=argparse.FileType('r'), help='read confusion matrix from this file', default='confusion-large.db')
parser.add_argument('-t', '--tagged', help='load tagged data from this directory', default='tagged/full')
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
args = parser.parse_args()

categories = []
for line in open('mp-dev_kit/categories.txt').readlines():
    assert int(line.strip().split()[1]) == len(categories)
    categories.append(re.sub('^/[a-z]/', '', line.strip().split()[0]))
cats = len(categories)

labels = {}
for line in open('mp-dev_kit/train.txt').readlines():
    name, label = line.strip().split()
    labels[name] = int(label)

filenames = [line.strip() for line in
        open('tagged/full/train.filenames.txt').readlines()]
cases = len(filenames)

predictions = numpy.memmap(args.confusion, dtype=numpy.float32, shape=(cases, cats), mode='r')
topindex = numpy.argsort(-predictions, axis=1)

for index in range(cases):
    top = topindex[index]
    topscore = predictions[index][top]
    correct = labels[filenames[index]]
    confusion = numpy.where(top == correct)[0][0]
    print("{} {} should be {}, was {} {}, {} {}, {} {}, {} {}, {} {}".format(
        confusion,
        filenames[index],
        categories[labels[filenames[index]]],
        categories[top[0]],
        topscore[0],
        categories[top[1]],
        topscore[1],
        categories[top[2]],
        topscore[2],
        categories[top[3]],
        topscore[3],
        categories[top[4]],
        topscore[4],
    ))
