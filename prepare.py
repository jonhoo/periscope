#!/usr/bin/env python3

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

def dir2nd(directory, into):
    N = 0
    for root, dirs, files in os.walk(directory, followlinks=True):
        for img in files:
            N += 1

    # Optionally limit length of training data for slow machines
    if args.limit > 0 and N > args.limit:
        N = args.limit

    i = 0
    images = numpy.memmap(into, dtype=numpy.float32, mode='w+', shape=(N, 3, 128, 128))
    for root, dirs, files in os.walk(directory, followlinks=True):
        for img in files:
            if i >= N:
                break

            images[i] = numpy.transpose(scipy.ndimage.imread(os.path.join(root, img)), [2, 0, 1]) / 255.0

            i += 1
            if (i+1) % 1000 == 0:
                print('Processed {} out of {} files'.format(i+1, N), file=sys.stderr)

    del images
    return N

print("Extracting training images...")
N = dir2nd(os.path.join(args.images, "train"), os.path.join(args.outdir, "train.images.db"))

print("Extracting validation images...")
dir2nd(os.path.join(args.images, "val"), os.path.join(args.outdir, "val.images.db"))

print("Extracting test images...")
dir2nd(os.path.join(args.images, "test"), os.path.join(args.outdir, "test.images.db"))

print("Extracting image labels...")
i = 0
categories = {}
labels = numpy.memmap(os.path.join(args.outdir, "train.labels.db"), dtype=numpy.int32, mode='w+', shape=(N, ))
for root, dirs, files in os.walk(os.path.join(args.images, "train"), followlinks=True):
    for img in files:
        if i >= N:
            break

        category = os.path.basename(root)
        if category not in categories:
            print("found category {}".format(category))
            label = len(categories)
            categories[category] = label

        labels[i] = categories[category]
        if (i+1) % 10000 == 0:
            print('Labelled {} out of {} images'.format(i+1, N), file=sys.stderr)
        i += 1
del labels
