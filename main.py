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
import re
import random
import os
import os.path

from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import rectify, softmax

Conv2DLayer = lasagne.layers.Conv2DLayer
MaxPool2DLayer = lasagne.layers.MaxPool2DLayer
if theano.config.device.startswith("gpu"):
    import lasagne.layers.dnn
    # Force GPU implementations if a GPU is available.
    # Do not know why Theano is not selecting these impls
    # by default as advertised.
    if theano.sandbox.cuda.dnn.dnn_available():
        Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
        MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer

parser = argparse.ArgumentParser()
parser.add_argument('tagged', help='path to directory containing prepared files')
parser.add_argument('-m', '--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('-b', '--batchsize', type=int, help='size of each mini batch', default=256)
parser.add_argument('-s', '--batch-stop', type=int, help='stop after this many batches each epoch', default=0)
parser.add_argument('-e', '--epoch-stop', type=int, help='stop after this many epochs', default=0)
parser.add_argument('-o', '--outdir', help='store trained network state in this directory', default='out')
parser.add_argument('--limit', type=int, help='limit analyses to this many images', default=None)
parser.add_argument('--labels', help='record test set predictions to this file', action='store_true')
parser.set_defaults(labels=False)
parser.add_argument('--no-plot', help='skip the plot', action='store_false')
parser.set_defaults(plot=True)
parser.add_argument('--confusion', help='compute confusion stats', action='store_true')
parser.set_defaults(confusion=False)
parser.add_argument('--response', help='compute response region', action='store_true')
parser.set_defaults(response=False)
parser.add_argument('-v', '--verbose', action='count')
args = parser.parse_args()

imsz = 128
cropsz = 117

section("Setup")
task("Loading data")
subtask("Loading training set")
y_train = numpy.memmap(os.path.join(args.tagged, "train.labels.db"), dtype=numpy.int32, mode='r')
X_train = numpy.memmap(os.path.join(args.tagged, "train.images.db"), dtype=numpy.float32, mode='r', shape=(len(y_train), 3, imsz, imsz))
cats = numpy.max(y_train)+1
subtask("Loading validation set")
y_val = numpy.memmap(os.path.join(args.tagged, "val.labels.db"), dtype=numpy.int32, mode='r')
X_val = numpy.memmap(os.path.join(args.tagged, "val.images.db"), dtype=numpy.float32, mode='r', shape=(len(y_val), 3, imsz, imsz))

if args.labels:
    subtask("Loading test set")
    y_test = numpy.memmap(os.path.join(args.tagged, "test.labels.db"), dtype=numpy.int32, mode='r')
    X_test = numpy.memmap(os.path.join(args.tagged, "test.images.db"), dtype=numpy.float32, mode='r', shape=(len(y_test), 3, imsz, imsz))

task("Building model and compiling functions")
# create Theano variables for input and target minibatch
learning_rates = numpy.logspace(-1.5, -4, 30, dtype=theano.config.floatX)
learning_rates_var = theano.shared(learning_rates)
learning_rate = theano.shared(learning_rates[0])
epochi = T.iscalar('e')
input_var = T.tensor4('X')
target_var = T.ivector('y')

# parameters
flip_var = T.iscalar('f')
crop_var = T.ivector('c') # ycrop, xcrop
center = numpy.zeros((2,), dtype=numpy.int32)
center.fill(numpy.floor((imsz - cropsz)/2))

# crop+flip
cropped = input_var[:, :, crop_var[0]:crop_var[0]+cropsz, crop_var[1]:crop_var[1]+cropsz]
prepared = cropped[:,:,:,::flip_var]

# create a small convolutional neural network
network = lasagne.layers.InputLayer((args.batchsize, 3, cropsz, cropsz), prepared)
# 1st
network = Conv2DLayer(network, 64, (8, 8), stride=2, nonlinearity=rectify)
network = BatchNormLayer(network, nonlinearity=rectify)
network = MaxPool2DLayer(network, (3, 3), stride=2)
# 2nd
network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same')
network = BatchNormLayer(network, nonlinearity=rectify)
network = MaxPool2DLayer(network, (3, 3), stride=2)
# 3rd
network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same')
network = BatchNormLayer(network, nonlinearity=rectify)
network = MaxPool2DLayer(network, (3, 3), stride=2)
# 4th
network = lasagne.layers.DenseLayer(network, 512)
network = BatchNormLayer(network, nonlinearity=rectify)
# 5th
network = lasagne.layers.DenseLayer(network, cats, nonlinearity=softmax)

# Output
prediction = lasagne.layers.get_output(network)

# create loss function
from lasagne.regularization import regularize_network_params, l2, l1
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
loss += regularize_network_params(network, l2) * 1e-3

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
saveparams = lasagne.layers.get_all_params(network)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=args.momentum)
updates[learning_rate] = learning_rates_var[epochi]
subtask("parameter count {} ({} trainable) in {} arrays".format(
        lasagne.layers.count_params(network),
        lasagne.layers.count_params(network, trainable=True),
        len(saveparams)))

# compile training function that updates parameters and returns training loss
train_fn = theano.function([epochi, flip_var, crop_var, input_var, target_var], loss, updates=updates)

# Create a loss expression for validation/testing. The crucial difference here
# is that we do a deterministic forward pass through the network, disabling
# dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_1_acc = T.mean(lasagne.objectives.categorical_accuracy(test_prediction, target_var, top_k=1))
test_5_acc = T.mean(lasagne.objectives.categorical_accuracy(test_prediction, target_var, top_k=5))

# compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var, theano.Param(flip_var, default=1), theano.Param(crop_var, default=center)], [test_loss, test_1_acc, test_5_acc])

# a function for test output
top5_pred = T.argsort(test_prediction, axis=1)[:, -5:]
test_fn = theano.function([input_var, theano.Param(flip_var, default=1), theano.Param(crop_var, default=center)], [top5_pred])
debug_fn = theano.function([input_var, theano.Param(flip_var, default=1), theano.Param(crop_var, default=center)], test_prediction)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, test=False):
    assert len(inputs) == len(targets)
    end = len(inputs)
    if shuffle:
        start = random.randrange(end)
        steps = [n % end for n in range(start, end + start, batchsize)]
        random.shuffle(steps)
    else:
        steps = range(0, end, batchsize)
    for start_idx in steps:
        if shuffle and start_idx + batchsize > end:
            # Handle wraparound case
            e1 = slice(start_idx, end)
            e2 = slice(0, (start_idx + batchsize) % end)
            yield (numpy.concatenate([inputs[e1], inputs[e2]]),
                   numpy.concatenate([targets[e1], targets[e2]]))
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

training = []
validation = []
end = len(learning_rates)
if args.epoch_stop != 0 and args.epoch_stop < end:
    end = args.epoch_stop

if args.plot:
    import matplotlib
    matplotlib.use('Agg') # avoid the need for X

def replot():
    if not args.plot:
        return
    global training
    global validation
    if len(validation) == 0:
        return

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="ticks", color_codes=True)

    fig = plt.figure()
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_err = fig.add_subplot(1, 2, 2)

    # styles
    ax_loss.grid(True)
    ax_err.grid(True)
    #ax_loss.set_yscale('log')
    #ax_err.set_yscale('log')

    # limits
    global end
    ax_loss.set_xlim(0, end)
    ax_err.set_xlim(0, end)
    ax_err.set_ylim(0, 1)
    #ax_err.set_ylim(1e-5, 1)

    # plot loss
    xend = len(training)+1
    ax_loss.plot(range(1, xend), [dp[0] for dp in training], 'b', marker='o', markersize=4)
    xend = len(validation)+1
    ax_loss.plot(range(1, xend), [dp[0] for dp in validation], 'r--', marker='o', markersize=4)
    ax_loss.legend(['Training loss', 'Validation loss'])
    ax_loss.set_title('Model loss')

    # plot error
    xend = len(training)+1
    ax_err.plot(range(1, xend), [1-dp[1] for dp in training], 'b', marker='o', markersize=4)
    ax_err.plot(range(1, xend), [1-dp[2] for dp in training], 'r', marker='o', markersize=4)
    xend = len(validation)+1
    ax_err.plot(range(1, xend), [1-dp[1] for dp in validation], 'y--', marker='s', markersize=4)
    ax_err.plot(range(1, xend), [1-dp[2] for dp in validation], 'm--', marker='s', markersize=4)
    ax_err.legend(['Training exact', 'Training top 5', 'Validation exact', 'Validation top 5'])
    ax_err.set_title('Match error')

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, dir=args.outdir) as fp:
        fig.savefig(fp, format='png', dpi=192)
        plt.close(fig)
        fp.close()
        os.rename(fp.name, os.path.join(args.outdir, 'plot.png'))

start = 0

def latest_cachefile(outdir):
    caches = [n for n in os.listdir(outdir) if re.match(r'^epoch-\d+\.mdl$', n)]
    if len(caches) == 0:
        return None
    caches.sort(key=lambda x: -int(re.match(r'^epoch-(\d+)\.mdl$', x).group(1)))
    return os.path.join(outdir, caches[0])

os.makedirs(args.outdir, exist_ok=True)
try:
    resumefile = latest_cachefile(args.outdir)
    if resumefile is None:
        raise EOFError
    with open(resumefile, 'rb') as lfile:
        section("Restoring state from {}".format(resumefile))
        lfile.seek(0)
        task("Loading format version")
        formatver = pickle.load(lfile)
        if type(formatver) != int:
            formatver = 0
            lfile.seek(0)
        subtask("using format {}".format(formatver))
        task("Loading parameters")
        state = pickle.load(lfile)
        task("Loading epoch information")
        epoch = pickle.load(lfile)
        training = pickle.load(lfile)
        validation = pickle.load(lfile)
        task("Restoring parameter values")
        fileparams = saveparams if formatver >= 1 else params
        assert len(fileparams) == len(state)
        for p, v in zip(fileparams, state):
            p.set_value(v)
        learning_rate.set_value(
               learning_rates[min(epoch, len(learning_rates) - 1)])
        subtask("Resuming at epoch {}".format(epoch+1))
        start = epoch+1
except EOFError:
    task("No model state stored; starting afresh")

section("Training")
# Finally, launch the training loop.
for epoch in range(start, end):
    replot()

    task("Starting training epoch {}".format(epoch))
    start_time = time.time()

    # How much work will we have to do?
    train_batches = len(range(0, len(X_train), args.batchsize))
    val_batches = len(range(0, len(X_val), args.batchsize))
    train_test_batches = val_batches

    # In each epoch, we do a pass over minibatches of the training data:
    train_loss = 0
    p = progress(train_batches)
    i = 1
    frame = numpy.zeros((2,), dtype=numpy.int32)
    for inp, res in iterate_minibatches(X_train, y_train, args.batchsize, shuffle=True):
        flip = numpy.random.randint(0, 2) and 1 or -1
        frame[0] = numpy.random.randint(0, imsz - cropsz)
        frame[1] = numpy.random.randint(0, imsz - cropsz)
        train_loss += train_fn(epoch, flip, frame, inp, res)
        p.update(i)
        i = i+1
        if args.batch_stop != 0 and i > args.batch_stop:
            p.update(train_batches)
            break

    # Only do forward pass on a subset of the training data
    subtask("Doing forward pass on training data (size: {})".format(len(X_val)))
    p = progress(train_test_batches)
    i = 0
    train_acc1 = 0
    train_acc5 = 0
    for inp, res in iterate_minibatches(X_train, y_train, args.batchsize, shuffle=True):
        i = i+1
        _, acc1, acc5 = val_fn(inp, res)
        p.update(i)
        train_acc1 += acc1
        train_acc5 += acc5
        if i == train_test_batches:
            break

    subtask("Doing forward pass on validation data (size: {})".format(len(X_val)))
    # Also do a validation data forward pass
    val_loss = 0
    val_acc1 = 0
    val_acc5 = 0
    p = progress(val_batches)
    i = 0
    for inp, res in iterate_minibatches(X_val, y_val, args.batchsize, shuffle=False):
        loss, acc1, acc5 = val_fn(inp, res)
        val_loss += loss
        val_acc1 += acc1
        val_acc5 += acc5
        i += 1
        p.update(i)

    # record performance
    training.append((train_loss/train_batches, train_acc1/train_test_batches, train_acc5/train_test_batches))
    validation.append((val_loss/val_batches, val_acc1/val_batches, val_acc5/val_batches))

    # store model state
    if args.outdir is not None:
        sfilename = os.path.join(args.outdir, 'epoch-%03d.mdl' % epoch)
        subtask("Storing trained parameters as {}".format(sfilename))
        with open(sfilename, 'wb+') as sfile:
            sfile.seek(0)
            sfile.truncate()
            formatver = 1
            pickle.dump(formatver, sfile)
            pickle.dump([p.get_value() for p in saveparams], sfile)
            pickle.dump(epoch, sfile)
            pickle.dump(training, sfile)
            pickle.dump(validation, sfile)

    # Then we print the results for this epoch:
    subtask(("Epoch results:" +
        " {:.2f}%/{:.2f}% (t1acc, v1acc)" +
        " {:.2f}%/{:.2f}% (t5acc, v5acc)").format(
        training[-1][1] * 100,
        validation[-1][1] * 100,
        training[-1][2] * 100,
        validation[-1][2] * 100,
    ))

replot()

if args.labels:
    section("Evaluation")
    task("Evaluating performance on test data set")
    efile = open(os.path.join(args.outdir, 'labels.db'), 'wb+')
    cases = len(X_test) if not args.limit else min(args.limit, len(X_test))
    pred_out = numpy.memmap(efile, dtype=numpy.int32, shape=(cases, 5))

    test_batches = len(range(0, cases, args.batchsize))
    p = progress(test_batches)
    i = 0
    for inp, res in iterate_minibatches(X_test, y_test, args.batchsize, shuffle=False):
        s = i * args.batchsize
        if s + args.batchsize > pred_out.shape[0]:
            inp = inp[:pred_out.shape[0] - s]
        pred_out[s:s+args.batchsize, :] = test_fn(inp)[0]
        i += 1
        p.update(i)
        if i == test_batches:
            break
    del pred_out
    efile.close()

def make_confusion_db(name, fname, X, Y):
    cfile = open(os.path.join(args.outdir, fname), 'wb+')
    cases = len(X) if not args.limit else min(args.limit, len(X))
    pred_out = numpy.memmap(
        cfile, dtype=numpy.float32, shape=(cases, cats), mode='w+')
    test_batches = len(range(0, cases, args.batchsize))
    p = progress(test_batches)
    i = 0
    accn = {}
    for inp, res in iterate_minibatches(X, Y, args.batchsize, shuffle=False):
        s = i * args.batchsize
        if s + args.batchsize > pred_out.shape[0]:
            inp = inp[:pred_out.shape[0] - s]
        pred_out[s:s+args.batchsize, :] = debug_fn(inp)
        i += 1
        p.update(i)
        topindex = numpy.argsort(-pred_out[s:s+args.batchsize], axis=1)
        for index in range(topindex.shape[0]):
            confusion = numpy.where(topindex[index] == res[index])[0][0]
            accn[confusion] = accn.get(confusion, 0) + 1
        if i >= test_batches:
            break
    correct = 0
    for index in range(10):
        correct += accn.get(index, 0)
        subtask("{} acc@{}: {:.2f}%".format(
                name,
                index + 1,
                100.0 * correct / cases))
    del pred_out
    cfile.close()

if args.confusion:
    section("Debugging")
    task("Evaluating confusion matrix on validation data")
    make_confusion_db('Validation set', 'val.confusion.db', X_val, y_val)
    task("Evaluating confusion matrix on training data set")
    make_confusion_db('Training set', 'train.confusion.db', X_train, y_train)

# Divides the image into 16x16 overlapping squares of 23x23 pixels, each
# offset 7 pixels from the previous
def make_response_probe(image):
    res = 16
    pix = 23
    st = 7
    result = numpy.tile(image, (res*res, 1, 1, 1))
    for x in range(res):
        for y in range(res):
            for c in range(3):
                avg = numpy.average(
                        result[x + y * res, c,
                               y*st:y*st+pix, x*st:x*st+pix])
                anti = (avg + 0.5) * numpy.ones([pix, pix])
                noise = numpy.random.normal(size=[pix-4, pix-4])
                box = anti
                box[2:pix-2,2:pix-2] += noise + 10
                box = box % 1
                # noise = (noise + anti) % 1
                # noise = avg * numpy.ones([pix, pix])
                # noise = anti * numpy.ones([pix, pix])
                result[x + y * res, c,
                       y*st:y*st+pix, x*st:x*st+pix] = box
    return result

def make_response_file(name, fname, X, Y):
    task("Evaluating response regions on %s" % name)
    assert args.batchsize == 256
    cases = len(X) if not args.limit else min(args.limit, len(X))
    rfile = open(os.path.join(args.outdir, fname), 'wb+')
    resp_out = numpy.memmap(rfile, dtype=numpy.float32,
            shape=(cases, 256), mode='w+')
    p = progress(cases)
    for index in range(cases):
        probe = make_response_probe(X[index])
        correct = Y[index]
        resp_out[index] = debug_fn(probe)[:,correct]
        p.update(index + 1)
    del resp_out
    rfile.close()

if args.response:
    section("Visualization")
    make_response_file('validation set', 'val.response.db', X_val, y_val)
    make_response_file('training set', 'train.response.db', X_train, y_train)
