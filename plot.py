#!/usr/bin/env python3

import argparse
import numpy
import sys
import os
import os.path
import tempfile

import matplotlib
matplotlib.use('Agg') # avoid the need for X
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    type=argparse.FileType('rb'),
                    help='path to .mdl to extract plot data from'
                    )
parser.add_argument('-l',
                    '--latex',
                    help='output in a latex-friendly format',
                    default=False,
                    action='store_true'
                    )
parser.add_argument('-a',
                    '--accuracy',
                    help='give only accuracy plot',
                    default=False,
                    action='store_true')
args = parser.parse_args()

training = []
validation = []

try:
    import pickle
    args.model.seek(0)
    formatver = pickle.load(args.model)
    if type(formatver) != int:
        formatver = 0
        args.model.seek(0)

    # add imports for unpickle to work
    import lasagne
    import theano
    pickle.load(args.model)  # state

    # the things we actually care about
    epoch = pickle.load(args.model)
    training = pickle.load(args.model)
    validation = pickle.load(args.model)
except EOFError:
    print("Invalid model given")
    sys.exit(1)

sns.set(style="ticks", color_codes=True)

fig = plt.figure()
if args.accuracy:
    ax_err = fig.gca()
else:
    ax_loss = fig.add_subplot(1, 2, 1)
    ax_err = fig.add_subplot(1, 2, 2)

# plot error
ax_err.grid(True)
ax_err.set_xlim(1, epoch+1)
ax_err.set_ylim(0, 1)
ax_err.yaxis.set_ticks(numpy.arange(0.0, 1.1, 0.1))
xend = len(training)+1
ax_err.plot(range(1, xend), [1-dp[1] for dp in training], 'b', marker='o', markersize=4)
ax_err.plot(range(1, xend), [1-dp[2] for dp in training], 'r', marker='o', markersize=4)
xend = len(validation)+1
ax_err.plot(range(1, xend), [1-dp[1] for dp in validation], 'y--', marker='s', markersize=4)
ax_err.plot(range(1, xend), [1-dp[2] for dp in validation], 'm--', marker='s', markersize=4)
ax_err.legend(['Training exact', 'Training top 5', 'Validation exact', 'Validation top 5'])
ax_err.set_title('Match error')

if not args.accuracy:
    # plot loss
    ax_loss.grid(True)
    ax_loss.set_xlim(1, epoch+1)
    xend = len(training)+1
    ax_loss.plot(range(1, xend), [dp[0] for dp in training], 'b', marker='o', markersize=4)
    xend = len(validation)+1
    ax_loss.plot(range(1, xend), [dp[0] for dp in validation], 'r--', marker='o', markersize=4)
    ax_loss.legend(['Training loss', 'Validation loss'])
    ax_loss.set_title('Model loss')

fig.savefig(sys.stdout, format='png', dpi=192)
plt.close(fig)
