#!/usr/bin/env python3

from progressbar import ProgressBar
from pretty import *
import argparse
import lasagne
import theano
import theano.tensor as T
import pickle
import numpy
import time

import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('predictions', type=argparse.FileType('rb+'), help='file with test predictions')
parser.add_argument('-n', '--names', action='store_true', help='output category names', default=False)
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
parser.add_argument('-t', '--tagged', help='load tagged data from this directory', default='tagged/full')
args = parser.parse_args()

filenames = [line.strip() for line in open(os.path.join(args.tagged,
        'test.filenames.txt')).readlines()]

categories = {}
if args.names:
    with open(os.path.join(args.devkit, "categories.txt"), 'r') as cmap:
        for line in cmap:
            c, ci = line.split(None, 1)
            categories[int(ci)] = os.path.basename(c)

predictions = numpy.memmap(args.predictions, dtype=numpy.int32, shape=(len(filenames), 5))

for i in range(len(predictions)):
    if args.names:
        cats = "\t".join([categories[ci] for ci in predictions[i]])
    else:
        cats = " ".join([str(c) for c in predictions[i]])
    print("{} {}".format(filenames[i], cats))
