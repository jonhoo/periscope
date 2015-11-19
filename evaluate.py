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
parser.add_argument('images', help='path to images/')
parser.add_argument('devkit', help="path to dev kit's data/")
parser.add_argument('predictions', type=argparse.FileType('rb+'), help='file with test predictions')
parser.add_argument('-n', '--names', action='store_true', help='output category names', default=False)
args = parser.parse_args()

images = []
for root, dirs, files in os.walk(os.path.join(args.images, "test"), followlinks=True):
    for img in files:
        images.append(os.path.relpath(os.path.join(root, img), args.images))

categories = {}
if args.names:
    with open(os.path.join(args.devkit, "categories.txt"), 'r') as cmap:
        for line in cmap:
            c, ci = line.split(None, 1)
            categories[int(ci)] = os.path.basename(c)

predictions = numpy.memmap(args.predictions, dtype=numpy.int32, shape=(len(images), 5))

for i in range(len(predictions)):
    if args.names:
        cats = "\t".join([categories[ci] for ci in predictions[i]])
    else:
        cats = " ".join([str(c) for c in predictions[i]])
    print("{} {}".format(images[i], cats))
