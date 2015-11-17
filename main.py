#!/usr/bin/env python3

from progressbar import ProgressBar
from pretty import *
import argparse
import lasagne
import theano
import theano.tensor as T
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

end = len(learning_rates)
if args.epoch_stop != 0:
    end = args.epoch_stop

section("Training")
# Finally, launch the training loop.
for epoch in range(start, end):
    task("Starting training epoch {}".format(epoch))
    start_time = time.time()

    # In each epoch, we do a pass over minibatches of the training data:
    train_err = 0
    train_batches = len(range(0, len(X_train) - args.batchsize + 1, args.batchsize))
    p = ProgressBar(max_value = train_batches).start()
    i = 1
    for batch in iterate_minibatches(X_train, y_train, args.batchsize, shuffle=True):
        train_err += train_fn(i-1, batch[0], batch[1])
        p.update(i)
        i = i+1
        if args.batch_stop != 0 and i > args.batch_stop:
            break

    subtask("Doing forward pass on validation data (size: {})".format(len(X_val)))
    # Also do a validation data forward pass
    val_err = 0
    val_acc1 = 0
    val_acc5 = 0
    val_batches = len(range(0, len(X_val) - args.batchsize + 1, args.batchsize))
    p = ProgressBar(max_value = val_batches).start()
    i = 1
    for batch in iterate_minibatches(X_val, y_val, args.batchsize, shuffle=False):
        err, acc1, acc5 = val_fn(batch[0], batch[1])
        val_err += err
        val_acc1 += acc1
        val_acc5 += acc5
        p.update(i)
        i = i+1

    # Then we print the results for this epoch:
    subtask("Epoch results: {:.6f}/{:.6f}/{:.2f}%/{:.2f}% (tloss, vloss, v1acc, v5acc)".format(
        train_err / train_batches,
        val_err / val_batches,
        val_acc1 / val_batches * 100,
        val_acc5 / val_batches * 100
    ))


section("Evaluation")
# After training, we compute and print the test error:
task("Evaluating performance on test data set")
test_err = 0
test_acc1 = 0
test_acc5 = 0
test_batches = len(range(0, len(X_test) - args.batchsize + 1, args.batchsize))
p = ProgressBar(max_value = test_batches).start()
i = 1
for batch in iterate_minibatches(X_test, y_test, args.batchsize, shuffle=False):
    err, acc1, acc5 = val_fn(batch[0], batch[1])
    test_err += err
    test_acc1 += acc1
    test_acc5 += acc5
    p.update(i)
    i = i+1

print(colored(" ==> Final results: {:.6f} loss, {:.2f}% top-1 accuracy, {:.2f}% top-5 accuracy <== ".format(
    test_err / test_batches,
    test_acc1 / test_batches * 100,
    test_acc5 / test_batches * 100
), "green", attrs=["bold"]))
