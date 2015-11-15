#!/usr/bin/env python3

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
parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=100)
parser.add_argument('-r', '--reserve', type=float, help='percentage of samples to reserve for validation and testing', default=5)
parser.add_argument('-l', '--learning', type=float, help='learning rate', default=0.01)
parser.add_argument('-m', '--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('-b', '--batchsize', type=int, help='size of each mini batch', default=100)
parser.add_argument('-v', '--verbose', action='count')
args = parser.parse_args()

print("Loading data...")
y_train = numpy.memmap(os.path.join(args.tagged, "train.labels.db"), dtype=numpy.int32, mode='r')
X_train = numpy.memmap(os.path.join(args.tagged, "train.images.db"), dtype=numpy.float32, mode='r', shape=(len(y_train), 3, 128, 128))

# use X% of data for validation, and X% for testing
reserved = int(args.reserve * len(y_train) / 100.0)
X_train, X_val, X_test = X_train[:-2*reserved], X_train[-2*reserved:-reserved], X_train[-reserved:]
y_train, y_val, y_test = y_train[:-2*reserved], y_train[-2*reserved:-reserved], y_train[-reserved:]

print("Building model and compiling functions...")
# create Theano variables for input and target minibatch
input_var = T.tensor4('X')
target_var = T.ivector('y')

# create a small convolutional neural network
from lasagne.nonlinearities import leaky_rectify, softmax
network = lasagne.layers.InputLayer((None, 3, 128, 128), input_var)
network = lasagne.layers.Conv2DLayer(network, 64, (8, 8),
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Conv2DLayer(network, 32, (8, 8),
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Pool2DLayer(network, (4, 4), stride=4, mode='max')
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    64, nonlinearity=leaky_rectify,
                                    W=lasagne.init.Orthogonal())
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    5, nonlinearity=softmax)

# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=args.learning, momentum=args.momentum)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)

# Create a loss expression for validation/testing. The crucial difference here
# is that we do a deterministic forward pass through the network, disabling
# dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

# compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

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

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(args.epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    nbatches = len(range(0, len(X_train) - args.batchsize + 1, args.batchsize))
    for batch in iterate_minibatches(X_train, y_train, args.batchsize, shuffle=True):
        started = time.time()
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1
        if args.verbose != 0:
            print("Batch {} of {} took {:.3f}s".format(
                train_batches, nbatches, time.time() - started))

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, args.batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, args.epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

# After training, we compute and print the test error:
test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(X_test, y_test, args.batchsize, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
