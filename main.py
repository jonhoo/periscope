#!/usr/bin/env python3

from progressbar import ProgressBar
from pretty import *
import argparse
import lasagne
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import pickle
import numpy
import time

import os
import os.path

from lasagne.layers.normalization import LocalResponseNormalization2DLayer
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
parser.add_argument('-c', '--cache', type=argparse.FileType('ab+'), help='store/resume network state in/from this file', default=None)
parser.add_argument('-p', '--plot', type=argparse.FileType('ab+'), help='plot network performance to this png file', default=None)
parser.add_argument('-l', '--labels', type=argparse.FileType('wb+'), help='record test set predictions to this file', default=None)
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

if args.labels is not None:
    subtask("Loading test set")
    y_test = numpy.memmap(os.path.join(args.tagged, "test.labels.db"), dtype=numpy.int32, mode='r')
    X_test = numpy.memmap(os.path.join(args.tagged, "test.images.db"), dtype=numpy.float32, mode='r', shape=(len(y_test), 3, imsz, imsz))

task("Building model and compiling functions")
# create Theano variables for input and target minibatch
learning_rates = numpy.logspace(-1.7, -4, 60, dtype=theano.config.floatX)
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
prepared = ifelse(T.eq(flip_var, 1), cropped[:,:,:,::-1], cropped)

# create a small convolutional neural network
network = lasagne.layers.InputLayer((None, 3, cropsz, cropsz), prepared)
# 1st
network = Conv2DLayer(network, 64, (8, 8), stride=2, nonlinearity=rectify)
network = LocalResponseNormalization2DLayer(network, n=5, k=1, beta=0.75, alpha=0.0001/5)
network = MaxPool2DLayer(network, (3, 3), stride=2)
# 2nd
network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same', nonlinearity=rectify)
network = LocalResponseNormalization2DLayer(network, n=5, k=1, beta=0.75, alpha=0.0001/5)
network = MaxPool2DLayer(network, (3, 3), stride=2)
# 3rd
network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', nonlinearity=rectify)
network = LocalResponseNormalization2DLayer(network, n=5, k=1, beta=0.75, alpha=0.0001/5)
network = MaxPool2DLayer(network, (3, 3), stride=2)
# 4th
network = lasagne.layers.DenseLayer(network, 512, nonlinearity=rectify)
network = lasagne.layers.DropoutLayer(network)
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
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=args.momentum)
updates[learning_rate] = learning_rates_var[epochi]

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
val_fn = theano.function([flip_var, crop_var, input_var, target_var], [test_loss, test_1_acc, test_5_acc])

# and a final one for test output
top5_pred = T.argsort(test_prediction, axis=1)[:, -5:]
test_fn = theano.function([flip_var, crop_var, input_var], [top5_pred])

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, test=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

training = []
validation = []
end = len(learning_rates)
if args.epoch_stop != 0:
    end = args.epoch_stop

if args.plot is not None:
    args.plot.close()
    import matplotlib
    matplotlib.use('Agg') # avoid the need for X

def replot():
    if args.plot is None:
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
    ax_loss.set_yscale('log')
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
    ax_loss.plot(range(1, xend), [dp[0] for dp in validation], 'r--', marker='o', markersize=4)
    ax_loss.legend(['Training loss', 'Validation loss'])
    ax_loss.set_title('Model loss')

    # plot error
    ax_err.plot(range(1, xend), [1-dp[1] for dp in training], 'b', marker='o', markersize=4)
    ax_err.plot(range(1, xend), [1-dp[2] for dp in training], 'r', marker='o', markersize=4)
    ax_err.plot(range(1, xend), [1-dp[1] for dp in validation], 'y--', marker='s', markersize=4)
    ax_err.plot(range(1, xend), [1-dp[2] for dp in validation], 'm--', marker='s', markersize=4)
    ax_err.legend(['Training exact', 'Training top 5', 'Validation exact', 'Validation top 5'])
    ax_err.set_title('Match error')

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(args.plot.name)) as fp:
        fig.savefig(fp, format='png', dpi=192)
        plt.close(fig)
        fp.close()
        os.rename(fp.name, args.plot.name)

start = 0
if args.cache is not None:
    try:
        section("Restoring state")
        args.cache.seek(0)
        task("Loading parameters")
        state = pickle.load(args.cache)
        task("Loading epoch information")
        epoch = pickle.load(args.cache)
        training = pickle.load(args.cache)
        validation = pickle.load(args.cache)
        task("Restoring parameter values")
        for p, v in zip(params, state):
            p.set_value(v)
        learning_rate.set_value(learning_rates[epoch])
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
    train_batches = len(range(0, len(X_train) - args.batchsize + 1, args.batchsize))
    val_batches = len(range(0, len(X_val) - args.batchsize + 1, args.batchsize))
    train_test_batches = val_batches

    # In each epoch, we do a pass over minibatches of the training data:
    train_loss = 0
    p = progress(train_batches)
    i = 1
    frame = numpy.zeros((2,), dtype=numpy.int32)
    for batch in iterate_minibatches(X_train, y_train, args.batchsize, shuffle=True):
        flip = numpy.random.randint(0, 2)
        frame[0] = numpy.random.randint(0, imsz - cropsz)
        frame[1] = numpy.random.randint(0, imsz - cropsz)
        train_loss += train_fn(epoch, flip, frame, batch[0], batch[1])
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
    for batch in iterate_minibatches(X_train, y_train, args.batchsize, shuffle=True):
        i = i+1
        _, acc1, acc5 = val_fn(0, center, batch[0], batch[1])
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
    i = 1
    for batch in iterate_minibatches(X_val, y_val, args.batchsize, shuffle=False):
        loss, acc1, acc5 = val_fn(0, center, batch[0], batch[1])
        val_loss += loss
        val_acc1 += acc1
        val_acc5 += acc5
        p.update(i)
        i = i+1

    # record performance
    training.append((train_loss/train_batches, train_acc1/train_test_batches, train_acc5/train_test_batches))
    validation.append((val_loss/val_batches, val_acc1/val_batches, val_acc5/val_batches))

    # store model state
    if args.cache is not None:
        subtask("Storing trained parameters")
        args.cache.seek(0)
        args.cache.truncate()
        pickle.dump([p.get_value() for p in params], args.cache)
        pickle.dump(epoch, args.cache)
        pickle.dump(training, args.cache)
        pickle.dump(validation, args.cache)

    # Then we print the results for this epoch:
    subtask("Epoch results: {:.2f}%/{:.2f}% (t5acc, v5acc)".format(
        training[-1][2] * 100,
        validation[-1][2] * 100,
    ))

replot()

if args.labels is not None:
    section("Evaluation")
    task("Evaluating performance on test data set")
    pred_out = numpy.memmap(args.labels, dtype=numpy.int32, shape=(len(X_test), 5))

    test_batches = len(range(0, len(X_test) - args.batchsize + 1, args.batchsize))
    p = progress(test_batches)

    i = 1
    for batch in iterate_minibatches(X_test, y_test, args.batchsize, shuffle=False):
        s = (i-1)*args.batchsize
        pred_out[s:s+args.batchsize, :] = test_fn(0, center, batch[0])[0]
        p.update(i)
        i = i+1
    del pred_out
