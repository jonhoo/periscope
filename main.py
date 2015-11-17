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
parser.add_argument('tagged', help='path to directory containing prepared files')
parser.add_argument('-r', '--reserve', type=float, help='percentage of samples to reserve for validation and testing', default=5)
parser.add_argument('-m', '--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('-b', '--batchsize', type=int, help='size of each mini batch', default=256)
parser.add_argument('-s', '--batch-stop', type=int, help='stop after this many batches each epoch', default=0)
parser.add_argument('-e', '--epoch-stop', type=int, help='stop after this many epochs', default=0)
parser.add_argument('-c', '--cache', type=argparse.FileType('ab+'), help='store/resume network state in/from this file', default=None)
parser.add_argument('-p', '--plot', type=argparse.FileType('ab+'), help='plot network performance to this png file', default=None)
parser.add_argument('-v', '--verbose', action='count')
args = parser.parse_args()

section("Setup")
task("Loading data")
y_train = numpy.memmap(os.path.join(args.tagged, "train.labels.db"), dtype=numpy.int32, mode='r')
X_train = numpy.memmap(os.path.join(args.tagged, "train.images.db"), dtype=numpy.float32, mode='r', shape=(len(y_train), 3, 128, 128))
cats = numpy.max(y_train)+1

# use X% of data for validation, and X% for testing
reserved = int(args.reserve * len(y_train) / 100.0)
X_train, X_val, X_test = X_train[:-2*reserved], X_train[-2*reserved:-reserved], X_train[-reserved:]
y_train, y_val, y_test = y_train[:-2*reserved], y_train[-2*reserved:-reserved], y_train[-reserved:]

task("Building model and compiling functions")
# create Theano variables for input and target minibatch
learning_rates = numpy.logspace(-2, -4, 60, dtype=theano.config.floatX)
learning_rates_var = theano.shared(learning_rates)
learning_rate = theano.shared(learning_rates[0] * (1-args.momentum)) # see https://lasagne.readthedocs.org/en/latest/modules/updates.html#lasagne.updates.nesterov_momentum
epochi = T.iscalar('e')
input_var = T.tensor4('X')
target_var = T.ivector('y')

# create a small convolutional neural network
from lasagne.nonlinearities import leaky_rectify, softmax
network = lasagne.layers.InputLayer((None, 3, 128, 128), input_var)
# 1st
network = lasagne.layers.Conv2DLayer(network, 64, (8, 8), stride=2,
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.MaxPool2DLayer(network, (3, 3), stride=2)
# 2nd
network = lasagne.layers.Conv2DLayer(network, 96, (5, 5), stride=1, pad='same',
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.MaxPool2DLayer(network, (3, 3), stride=2)
# 3rd
network = lasagne.layers.Conv2DLayer(network, 128, (3, 3), stride=1, pad='same',
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.MaxPool2DLayer(network, (3, 3), stride=2)
# 4th
network = lasagne.layers.DenseLayer(network, 512, nonlinearity=leaky_rectify)
network = lasagne.layers.DropoutLayer(network)
# 5th
network = lasagne.layers.DenseLayer(network, cats, nonlinearity=softmax)

# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=args.momentum)
updates[learning_rate] = learning_rates_var[epochi]

# compile training function that updates parameters and returns training loss
train_fn = theano.function([epochi, input_var, target_var], loss, updates=updates)

# Create a loss expression for validation/testing. The crucial difference here
# is that we do a deterministic forward pass through the network, disabling
# dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_1_acc = T.mean(lasagne.objectives.categorical_accuracy(test_prediction, target_var))
test_5_acc = T.mean(T.any(T.eq(T.argsort(test_prediction, axis=1)[:, -5:], target_var.dimshuffle(0, 'x')), axis=1), dtype=theano.config.floatX)

# compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_1_acc, test_5_acc])

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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
    #ax_loss.set_yscale('log')
    #ax_err.set_yscale('log')

    # limits
    global end
    ax_loss.set_xlim(0, end)
    ax_err.set_xlim(0, end)
    ax_err.set_ylim(0, 1)
    #ax_err.set_ylim(1e-5, 1)

    # plot loss
    ax_loss.plot([dp for dp in training])
    ax_loss.plot([dp[0] for dp in validation])
    ax_loss.legend(['Training loss', 'Validation loss'])

    # plot error
    ax_err.plot([1-dp[1] for dp in validation])
    ax_err.plot([1-dp[2] for dp in validation])
    ax_err.legend(['Exact match error', 'Top 5 match error'])

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(args.plot.name)) as fp:
        plt.savefig(fp, format='png')
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
        learning_rate.set_value(learning_rates[epoch] * (1-args.momentum))
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

    # In each epoch, we do a pass over minibatches of the training data:
    train_loss = 0
    train_batches = len(range(0, len(X_train) - args.batchsize + 1, args.batchsize))
    p = ProgressBar(max_value = train_batches).start()
    i = 1
    for batch in iterate_minibatches(X_train, y_train, args.batchsize, shuffle=True):
        train_loss += train_fn(i-1, batch[0], batch[1])
        p.update(i)
        i = i+1
        if args.batch_stop != 0 and i > args.batch_stop:
            break

    subtask("Doing forward pass on validation data (size: {})".format(len(X_val)))
    # Also do a validation data forward pass
    val_loss = 0
    val_acc1 = 0
    val_acc5 = 0
    val_batches = len(range(0, len(X_val) - args.batchsize + 1, args.batchsize))
    p = ProgressBar(max_value = val_batches).start()
    i = 1
    for batch in iterate_minibatches(X_val, y_val, args.batchsize, shuffle=False):
        loss, acc1, acc5 = val_fn(batch[0], batch[1])
        val_loss += loss
        val_acc1 += acc1
        val_acc5 += acc5
        p.update(i)
        i = i+1

    # record performance
    training.append(train_loss / train_batches)
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
    subtask("Epoch results: {:.6f}/{:.6f}/{:.2f}%/{:.2f}% (tloss, vloss, v1acc, v5acc)".format(
        training[-1],
        validation[-1][0],
        validation[-1][1] * 100,
        validation[-1][2] * 100,
    ))

replot()

section("Evaluation")
# After training, we compute and print the test error:
task("Evaluating performance on test data set")
test_loss = 0
test_acc1 = 0
test_acc5 = 0
test_batches = len(range(0, len(X_test) - args.batchsize + 1, args.batchsize))
p = ProgressBar(max_value = test_batches).start()
i = 1
for batch in iterate_minibatches(X_test, y_test, args.batchsize, shuffle=False):
    loss, acc1, acc5 = val_fn(batch[0], batch[1])
    test_loss += loss
    test_acc1 += acc1
    test_acc5 += acc5
    p.update(i)
    i = i+1

print(colored(" ==> Final results: {:.6f} loss, {:.2f}% top-1 accuracy, {:.2f}% top-5 accuracy <== ".format(
    test_loss / test_batches,
    test_acc1 / test_batches * 100,
    test_acc5 / test_batches * 100
), "green", attrs=["bold"]))
