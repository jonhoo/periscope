#!/usr/bin/env python3

import argparse
import numpy
import re
import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tagged', help='load tagged data from this directory', default='tagged/full')
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
parser.add_argument('-o', '--outdir', help='store trained network state in this directory', default=None)
parser.add_argument('-n', '--network', help='name of network experiment', default='base')
parser.add_argument('-s', '--subset', help='train or val', default='train')
args = parser.parse_args()

if args.outdir is None:
    args.outdir = "exp-{}".format(args.network)


categories = []
for line in open(os.path.join(args.devkit, 'categories.txt')).readlines():
    assert int(line.strip().split()[1]) == len(categories)
    categories.append(re.sub('^/[a-z]/', '', line.strip().split()[0]))
cats = len(categories)

labels = {}
for line in open(os.path.join(
        args.devkit, '%s.txt' % args.subset)).readlines():
    name, label = line.strip().split()
    labels[name] = int(label)

filenames = [line.strip() for line in open(os.path.join(
        args.tagged, '%s.filenames.txt' % args.subset)).readlines()]
cases = len(filenames)

confusion_file = open(os.path.join(args.outdir,
    '%s.confusion.db' % args.subset), 'r')
predictions = numpy.memmap(
    confusion_file, dtype=numpy.float32, shape=(cases, cats), mode='r')
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
