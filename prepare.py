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
parser.add_argument('devkit', help="path to dev kit's data/")
parser.add_argument('outdir', help='path to directory in which to place serialized data files')
parser.add_argument('-c', '--categories', type=int, help='limit number of categories to sample', default=0)
parser.add_argument('-s', '--samples', type=int, help='limit number of images to sample per category', default=0)
args = parser.parse_args()

def dir2nd(directory, ignore_limit=False):
    global args

    # Count images
    N = sum([len(files) for _, _, files in os.walk(os.path.join(args.images, directory), followlinks=True)])

    # Grab label map
    Ncat = None
    Nipc = None # images per category
    labels = None
    try:
        with open(os.path.join(args.devkit, directory) + '.txt', 'r') as lmap:
            labels = dict(line.split(' ', 1) for line in lmap)
            # It'd be great if we just iterated over the filenames listed here
            # instead of traversing the file system as below, but this wouldn't
            # work because we don't have labels for the test images.
            Ncat = len(set(labels.values()))
            # We assume images are equally divided among categories
            Nipc = numpy.ceil(N / Ncat)
    except FileNotFoundError:
        pass

    # Optionally restrict number of files to parse
    limit = N
    if not ignore_limit and args.categories != 0 and args.samples != 0:
        limit = args.categories * args.samples
    elif args.categories != 0 and Nipc is not None:
        limit = args.categories * Nipc
    elif not ignore_limit and args.samples != 0:
        limit = Ncat * args.samples
    if limit < N:
        N = limit

    remap = numpy.arange(N)
    numpy.random.shuffle(remap)

    i = 0
    p = ProgressBar(max_value = N, redirect_stdout=True).start()
    imdb = numpy.memmap(os.path.join(args.outdir, directory) + '.images.db', dtype=numpy.float32, mode='w+', shape=(N, 3, 128, 128))
    lbdb = numpy.memmap(os.path.join(args.outdir, directory) + '.labels.db', dtype=numpy.int32, mode='w+', shape=(N, ))
    for root, dirs, files in os.walk(os.path.join(args.images, directory), followlinks=True):
        imgs = 0
        for img in files:
            impath = os.path.join(root, img)
            rel = os.path.relpath(impath, args.images)
            cat = None
            if labels is not None:
                cat = int(labels[rel])
            if cat is not None and args.categories != 0 and cat >= args.categories:
                # Skip categories after the limit. Note that we don't do the
                # "first N catgories", we do "lowest N categories". This is so
                # that the validation set, when limited, will be restricted to
                # the same categories as the training set is. For the test set,
                # which doesn't have labels, we (obviously) don't limit the
                # number of categories.
                continue

            imdb[remap[i]] = numpy.transpose(scipy.ndimage.imread(impath), [2, 0, 1]) / 255.0
            lbdb[remap[i]] = cat is None and -1 or cat

            i += 1
            p.update(i)

            imgs += 1
            # For the validation set, we ignore limits on the number of images
            # per category to extract, as it's fairly small anyway. Note that
            # there is no need for a check on args.samples != 0 here, because
            # if args.samples == 0, this will never trigger.
            if not ignore_limit and imgs == args.samples:
                break
    p.finish()

    del imdb
    del lbdb
    return (N, remap)

section("Dataset preparation")

task("Extracting training images")
dir2nd("train")

task("Extracting validation images")
dir2nd("val", True)

task("Extracting test images")
dir2nd("test", True)
