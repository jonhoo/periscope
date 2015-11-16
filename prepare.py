#!/usr/bin/env python3

from progressbar import ProgressBar
from pretty import *
import scipy.ndimage
import argparse
import os.path
import numpy
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('images', help='path to images/')
parser.add_argument('outdir', help='path to directory in which to place serialized data files')
parser.add_argument('-l', '--limit', type=int, help='limit number of images to sample', default=0)
args = parser.parse_args()

# Optionally limit length of training data for slow machines
MAX_PER_DIR=1000
limit_dirs = -1
if args.limit != 0:
    limit_dirs = numpy.ceil(args.limit / float(MAX_PER_DIR))

def dir2nd(directory, into, ignore_limit=False):
    N = 0
    for root, dirs, files in os.walk(directory, followlinks=True):
        for img in files:
            N += 1

    if not ignore_limit and args.limit != 0 and args.limit < N:
        N = args.limit

    remap = numpy.arange(N)
    numpy.random.shuffle(remap)

    i = 0
    dirs_left = limit_dirs
    p = ProgressBar(max_value = N, redirect_stdout=True).start()
    images = numpy.memmap(into, dtype=numpy.float32, mode='w+', shape=(N, 3, 128, 128))
    for root, dirs, files in os.walk(directory, followlinks=True):
        if not ignore_limit and dirs_left == 0:
            break

        imgs = 0
        for img in files:
            images[remap[i]] = numpy.transpose(scipy.ndimage.imread(os.path.join(root, img)), [2, 0, 1]) / 255.0

            i += 1
            p.update(i)

            imgs += 1
            if not ignore_limit and args.limit != 0 and imgs == MAX_PER_DIR:
                break

        if len(files) != 0:
            dirs_left -= 1
    p.finish()

    del images
    return (N, remap)

section("Dataset preparation")

#task("Extracting validation images")
#dir2nd(os.path.join(args.images, "val"), os.path.join(args.outdir, "val.images.db"), True)

#task("Extracting test images")
#dir2nd(os.path.join(args.images, "test"), os.path.join(args.outdir, "test.images.db"), True)

task("Preparing training dataset")
subtask("Extracting training images...")
N, remap = dir2nd(os.path.join(args.images, "train"), os.path.join(args.outdir, "train.images.db"))

i = 0
categories = {}
dirs_left = limit_dirs
subtask("Extracting image labels...")
p = ProgressBar(max_value = N, redirect_stdout=True).start()
labels = numpy.memmap(os.path.join(args.outdir, "train.labels.db"), dtype=numpy.int32, mode='w+', shape=(N, ))
for root, dirs, files in os.walk(os.path.join(args.images, "train"), followlinks=True):
    if dirs_left == 0:
        break

    imgs = 0
    for img in files:
        category = os.path.basename(root)
        if category not in categories:
            label = len(categories)
            print("found category {}, assigned id {}".format(category, label))
            categories[category] = label

        labels[remap[i]] = categories[category]

        i += 1
        p.update(i)

        imgs += 1
        if args.limit != 0 and imgs == MAX_PER_DIR:
            break

    if len(files) != 0:
        dirs_left -= 1
del labels
