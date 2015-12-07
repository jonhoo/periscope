#!/usr/bin/env python3

from progressbar import ProgressBar
from pretty import *
import argparse
import experiment
import lasagne
import theano
import theano.tensor as T
import pickle
import numpy
import time
import re
import random
import os
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tagged', help='path to directory containing prepared files', default='tagged/full')
parser.add_argument('-n', '--network', help='name of network experiment', nargs='+', required=True)
parser.add_argument('-m', '--model', type=argparse.FileType('rb'), help='trained model to evaluate', nargs='+', required=True)
parser.add_argument('-b', '--batchsize', type=int, help='size of each mini batch', default=256)
parser.add_argument('-s', '--set', help='image set to evaluate on', choices=['test', 'val'], default='test')
parser.add_argument('-l', '--labels', action='store_true', help='output category labels', default=False)
parser.add_argument('-d', '--devkit', help='devkit directory containing categories.txt', default='mp-dev_kit')
parser.add_argument('-c', '--combine', help='combine the output of multiple cropflips', default=False, action='store_true')
args = parser.parse_args()

assert len(args.network) == len(args.model)
if len(args.model) != 1 and not args.combine:
    import sys
    print("Cannot evaluate on multiple models without combination", file=sys.stderr)
    sys.exit(1)

imsz = 128
cropsz = 117

center = numpy.floor((imsz - cropsz)/2)
crops = [0, center, imsz - cropsz - 1]
flips = [False, True]

section("Setup")
task("Loading data")
subtask("Loading categories")
cats = numpy.max(numpy.memmap(os.path.join(args.tagged, "train.labels.db"), dtype=numpy.int32, mode='r'))+1
subtask("Loading {} set".format(args.set))
y_test = numpy.memmap(os.path.join(args.tagged, "{}.labels.db".format(args.set)), dtype=numpy.int32, mode='r')
X_test = numpy.memmap(os.path.join(args.tagged, "{}.images.db".format(args.set)), dtype=numpy.float32, mode='r', shape=(len(y_test), 3, imsz, imsz))

section("Constructing networks")
ni = 0
networks = []
for m in args.model:
    task(m.name)
    subtask("Building model and compiling functions")

    # create Theano variables for input and target minibatch
    input_var = T.tensor4('X')

    # input layer is always the same
    network = lasagne.layers.InputLayer((args.batchsize, 3, cropsz, cropsz), input_var)

    # import external network
    if args.network[ni] not in experiment.__dict__:
        print("No network {} found.".format(args.network[ni]))
        import sys
        sys.exit(1)

    # dispatch to user-defined network
    network = experiment.__dict__[args.network[ni]](network, cropsz, args.batchsize)

    # Last softmax layer is always the same
    from lasagne.nonlinearities import softmax
    network = lasagne.layers.DenseLayer(network, cats, nonlinearity=softmax)

    # create parameter update expressions
    params = lasagne.layers.get_all_params(network, trainable=True)
    saveparams = lasagne.layers.get_all_params(network)

    # Create an evaluation expression for testing.
    networks.append(theano.function([input_var], [lasagne.layers.get_output(network, deterministic=True)]))

    # Load model parameters
    subtask("Restoring state from {}".format(m.name))
    m.seek(0)
    formatver = pickle.load(m)
    if type(formatver) != int:
        formatver = 0
        m.seek(0)

    # Load parameters
    state = pickle.load(m)
    pickle.load(m) # epoch
    pickle.load(m) # training
    pickle.load(m) # validation
    # Restore parameter values
    fileparams = saveparams if formatver >= 1 else params
    assert len(fileparams) == len(state)
    for p, v in zip(fileparams, state):
        p.set_value(v)

    ni += 1

def iterate_minibatches(inputs):
    global args
    end = len(inputs)
    steps = range(0, end, args.batchsize)
    for start_idx in steps:
        yield inputs[slice(start_idx, start_idx + args.batchsize)]

section("Evaluation")
task("Evaluating performance on {} data set".format(args.set))
cases = len(X_test)
predictions = numpy.zeros((len(X_test), 5))

test_batches = len(range(0, cases, args.batchsize))
p = progress(test_batches)
i = 0
_preds = None
if args.combine:
    _preds = numpy.zeros((len(flips)*len(crops)*len(crops)*len(networks), args.batchsize, cats))

for inp in iterate_minibatches(X_test):
    s = i * args.batchsize
    if s + args.batchsize > predictions.shape[0]:
        inp = inp[:predictions.shape[0] - s]

    if not args.combine:
        # center crop
        predictions[s:s+len(inp), :] = numpy.argsort(networks[0](inp[:, :, center:center+cropsz, center:center+cropsz])[0])[:, -5:][:, ::-1]
    else:
        config = 0
        _preds.fill(0)
        for flip in flips:
            if flip:
                # flip once here instead of having to flip multiple times on the GPU
                inp = inp[:, :, :, ::-1]
            for xcrops in crops:
                if flip:
                    xcrop = slice(xcrops, xcrops+cropsz)
                else:
                    xcrop = slice(xcrops+cropsz-1, None if xcrops == 0 else xcrops-1, -1)

                for ycrop in crops:
                    cropped = inp[:, :, ycrop:ycrop+cropsz, xcrop]

                    for network in networks:
                        _preds[config, :len(inp), :] = network(cropped)[0]
                        config += 1

        # take median across configurations
        # pick top 5 categories
        # last category is highest probability
        predictions[s:s+len(inp), :] = numpy.argsort(numpy.median(_preds, axis=0))[:len(inp), -5:][:, ::-1]

    i += 1
    p.update(i)

filenames = [line.strip() for line in open(os.path.join(args.tagged,
        'test.filenames.txt')).readlines()]

categories = {}
if args.labels:
    with open(os.path.join(args.devkit, "categories.txt"), 'r') as cmap:
        for line in cmap:
            c, ci = line.split(None, 1)
            categories[int(ci)] = os.path.basename(c)

for i in range(len(predictions)):
    if args.labels:
        cats = "\t".join([categories[ci] for ci in predictions[i]])
    else:
        cats = " ".join([str(int(c)) for c in predictions[i]])
    print("{} {}".format(filenames[i], cats))

if args.set != 'test':
    top1 = numpy.mean(numpy.equal(predictions[:, 0], y_test))
    top5 = numpy.mean(numpy.any(numpy.equal(predictions[:, 0:5], y_test.reshape(-1, 1)), axis=1))
    task("Evaluation accuracy: exact: {}, top-5: {}".format(top1, top5))
