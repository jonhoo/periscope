#!/usr/bin/env python3

import argparse
import numpy
import re
import os
import os.path
from scipy import misc

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--response', type=argparse.FileType('r'), help='read response matrix from this file', default='response-large.db')
parser.add_argument('-t', '--tagged', help='load tagged data from this directory', default='tagged/full')
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
parser.add_argument('-i', '--images', help='path to images/', default='mp-data/images')
args = parser.parse_args()

categories = []
for line in open(os.path.join(args.devkit, 'categories.txt')).readlines():
    assert int(line.strip().split()[1]) == len(categories)
    categories.append(re.sub('^/[a-z]/', '', line.strip().split()[0]))
cats = len(categories)

labels = {}
for line in open(os.path.join(args.devkit, 'train.txt')).readlines():
    name, label = line.strip().split()
    labels[name] = int(label)

filenames = [line.strip() for line in
        open(os.path.join(args.tagged, 'train.filenames.txt')).readlines()]
cases = len(filenames)

response = numpy.memmap(args.response, dtype=numpy.float32, shape=(cases, 16, 16), mode='r')

index = 14
resp = response[index]
avg = numpy.average(resp)
std = numpy.std(resp)
respmax = avg + std
respmin = avg - std
resp = numpy.minimum(numpy.maximum((respmax - resp) / (respmax - respmin), 0), 1)
smeared = numpy.zeros([128, 128])
smearedd = numpy.ones([128, 128]) * 0.01
res = 16
pix = 38
st = 6
off = 0

def peg(ar, t1, t2):
    return numpy.clip((ar - t1) / (t2 - t1), 0, 1)

for x in range(res):
    for y in range(res):
        smeared[off+y*st:off+y*st+pix, off+x*st:off+x*st+pix] += resp[y][x]
        smearedd[off+y*st:off+y*st+pix, off+x*st:off+x*st+pix] += 1
r = peg(numpy.clip(numpy.divide(smeared, smearedd), 0, 1), 0.4, 0.6)
rpix = 255 * numpy.tile(r.reshape([128, 128, 1]), [1, 1, 3])


im = misc.imread(os.path.join(args.images, filenames[index]))
# im = numpy.ones([128,128,3]) * 255
seeim = numpy.clip(im - (255 - rpix), 0, 255).astype(numpy.uint8)
ignoreim = numpy.clip(im - rpix, 0, 255).astype(numpy.uint8)

misc.imsave('see.png', seeim)
misc.imsave('ignore.png', ignoreim)

