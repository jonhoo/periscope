#!/usr/bin/env python3

from progressbar import ProgressBar
from pretty import *
import argparse
import numpy
import re
import os
import os.path
from scipy import misc
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tagged', help='load tagged data from this directory', default='tagged/full')
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
parser.add_argument('-i', '--images', help='path to images/', default='mp-data/images')
parser.add_argument('-o', '--outdir', help='directory to save images', default='exp-large')
parser.add_argument('--serve', help='run http server', action='store_true')
parser.set_defaults(serve=False)
args = parser.parse_args()

# A clipping rectifier, pegging values less than t1 to 0 and more than t2 to 1.
def peg(ar, t1, t2):
    return numpy.clip((ar - t1) / (t2 - t1), 0, 1)

# regular python argsort, from stack overflow
def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

# Category labels
categories = []
for line in open(os.path.join(args.devkit, 'categories.txt')).readlines():
    assert int(line.strip().split()[1]) == len(categories)
    categories.append(re.sub('^/[a-z]/', '', line.strip().split()[0]))
cats = len(categories)


def create_response_images(name, subset):
    task("Generating images showing response regions for {}".format(name))
    labels = {}
    for line in open(os.path.join(args.devkit, '%s.txt' % subset)).readlines():
        name, label = line.strip().split()
        labels[name] = int(label)
    filenames = [line.strip() for line in open(os.path.join(args.tagged,
            '%s.filenames.txt' % subset)).readlines()]
    response = numpy.memmap(
            os.path.join(args.outdir, '%s.response.db' % subset),
            dtype=numpy.float32, mode='r')
    response.shape = (response.shape[0] / 256, 16, 16)
    cases = response.shape[0]
    p = progress(cases)
    for index in range(cases):
        # Smear the measured response in the way it was collected:
        # over the image with 16 overlapping blocks with a 6-pixel stride.
        # the blocks are made larger here to fill the entire 128px image,
        # and additionally a gaussian blur is added at the end.
        resp = response[index]
        avg = numpy.average(resp)
        std = numpy.std(resp)
        respmax = avg + std
        respmin = avg - std
        resp = numpy.minimum(
                numpy.maximum((respmax - resp) /
                (respmax - respmin + 1e-6), 0), 1)
        smeared = numpy.zeros([128, 128])
        smearedd = numpy.ones([128, 128]) * 0.01
        # Some geometric parameters
        res = 16
        pix = 23
        st = 7
        for x in range(res):
            for y in range(res):
                smeared[y*st:y*st+pix, x*st:x*st+pix] += resp[y][x]
                smearedd[y*st:y*st+pix, x*st:x*st+pix] += 1
        r = peg(numpy.clip(gaussian_filter(
                numpy.divide(smeared, smearedd), sigma=st), 0, 1), 0.4, 0.6)
        # now apply the response to create two images.
        r3 = numpy.tile(r.reshape([128, 128, 1]), [1, 1, 3])
        im = misc.imread(os.path.join(args.images, filenames[index]))
        # the "see" image shows where the response was high.
        seeim = numpy.multiply(im, r3)
        seename = os.path.join(args.outdir, 'resp', 'see', filenames[index])
        os.makedirs(os.path.dirname(seename), exist_ok=True)
        misc.imsave(seename, seeim)
        # the "ignore" image shows where the response was low.
        ignoreim = numpy.multiply(im, 1 - r3)
        ignorename = os.path.join(
                args.outdir, 'resp', 'ignore', filenames[index])
        os.makedirs(os.path.dirname(ignorename), exist_ok=True)
        misc.imsave(ignorename, ignoreim)
        # the "good" image blurs the ignore image and shows the see image
        blurred = numpy.zeros(im.shape)
        for c in range(3):
            blurred[:,:,c] = gaussian_filter(im[:,:,c], sigma=12)
        blurignore = numpy.multiply(blurred, 1 - r3)
        goodim = blurignore + seeim
        goodname = os.path.join(args.outdir, 'resp', 'good', filenames[index])
        os.makedirs(os.path.dirname(goodname), exist_ok=True)
        misc.imsave(goodname, goodim)

        blursee = numpy.multiply(blurred, r3)
        badim = blursee + ignoreim
        badname = os.path.join(args.outdir, 'resp', 'bad', filenames[index])
        os.makedirs(os.path.dirname(badname), exist_ok=True)
        misc.imsave(badname, badim)
        p.update(index + 1)

def create_eval_html(name, subset):
    html = [
        '<!doctype html>',
        '<html>',
        '<body>'
    ]
    labels = {}
    for line in open(os.path.join(args.devkit, '%s.txt' % subset)).readlines():
        name, label = line.strip().split()
        labels[name] = int(label)
    filenames = [line.strip() for line in open(os.path.join(args.tagged,
            '%s.filenames.txt' % subset)).readlines()]
    predictions = numpy.memmap(os.path.join(args.outdir,
            '%s.confusion.db' % subset), dtype=numpy.float32, mode='r')
    cases = int(predictions.shape[0] / cats)
    predictions.shape = (cases, cats)
    topindex = numpy.argsort(-predictions, axis=1)
    confusion = []
    for index in range(cases):
        top = topindex[index]
        correct = labels[filenames[index]]
        confusion.append(numpy.where(top == correct)[0][0])
    worstfirst = reversed(argsort(confusion))
    for index in worstfirst:
        top = topindex[index]
        correct = labels[filenames[index]]
        score = predictions[index]
        html.append('<p><img src="resp/good/' + filenames[index] + '">')
        html.append('<img src="resp/bad/' + filenames[index] + '">')
        html.append('<br>{} should be {} {}, was {} {}.  Confusion {}.'.format(
                filenames[index],
                categories[correct],
                score[correct],
                categories[top[0]],
                score[top[0]],
                confusion[index]
        ))
    html.extend(['</body>', '</html>'])
    hfilename = os.path.join(args.outdir, '%s.eval.html' % subset)
    with open(hfilename, 'w') as htmlfile:
        htmlfile.write('\n'.join(html))


create_eval_html("validation set", "val")
create_eval_html("training set", "train")
create_response_images("validation set", "val")
create_response_images("training set", "train")

if args.serve:
    from http.server import HTTPServer
    from http.server import SimpleHTTPRequestHandler
    os.chdir(args.outdir)
    PORT = 8000
    httpd = HTTPServer(('', PORT), SimpleHTTPRequestHandler)
    print("serving at port", PORT)
    httpd.serve_forever()
