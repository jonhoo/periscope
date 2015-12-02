#!/usr/bin/env python3

import argparse
import numpy
import sys
import re
import os
import os.path
import tempfile

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    type=argparse.FileType('rb'),
                    nargs='+',
                    help='path to .mdl to extract plot data from'
                    )
parser.add_argument('-l',
                    '--latex',
                    help='output in a latex-friendly format',
                    default=False,
                    action='store_true'
                    )
parser.add_argument('-f', '--format', help='image format', default=None)
args = parser.parse_args()

if args.format is not None:
    matplotlib.use('Agg') # avoid the need for X

maxe = 0
training = []
validation = []

for model in args.model:
    try:
        import pickle
        model.seek(0)
        formatver = pickle.load(model)
        if type(formatver) != int:
            formatver = 0
            model.seek(0)

        # add imports for unpickle to work
        import lasagne
        import theano
        pickle.load(model)  # state

        # the things we actually care about
        epoch = pickle.load(model)
        if epoch > maxe:
            maxe = epoch

        training.append(pickle.load(model))
        validation.append(pickle.load(model))
    except EOFError:
        print("Model {} is invalid".format(model.name))
        sys.exit(1)

sns.set(style="ticks", color_codes=True)

fig = plt.figure()
ax_err = fig.gca()

# plot error
ax_err.grid(True)
ax_err.set_xlim(1, maxe+1)
ax_err.set_ylim(0, 1)
ax_err.yaxis.set_ticks(numpy.arange(0.0, 1.1, 0.1))
ax_err.set_title('Match error')

tlegends = []
for i in range(len(training)):
    model = re.sub('\.mdl$', '', args.model[i].name)
    model = re.sub(os.path.sep + 'epoch-\d+$', '', model)
    if os.path.sep in model:
        model = os.path.basename(model)

    # exact is s, top5 is o
    # training is '--', validation is ''

    xend = len(training[i])+1
    c = ax_err.plot(range(1, xend), [1-dp[1] for dp in training[i]], '--', marker='s', markersize=4)
    c = c[0].get_color()
    tlegends.append('{} training exact'.format(model))
    ax_err.plot(range(1, xend), [1-dp[2] for dp in training[i]], '--', color=c, marker='o', markersize=4)
    tlegends.append('{} training top 5'.format(model))

    xend = len(validation[i])+1
    ax_err.plot(range(1, xend), [1-dp[1] for dp in validation[i]], '', color=c, marker='s', markersize=4)
    tlegends.append('{} validation exact'.format(model))
    ax_err.plot(range(1, xend), [1-dp[2] for dp in validation[i]], '', color=c, marker='o', markersize=4)
    tlegends.append('{} validation top 5'.format(model))

ax_err.legend(tlegends, ncol=len(training), prop={'size':8})

if args.format is None:
    plt.show(fig)
else:
    fig.savefig(sys.stdout, format=args.format, dpi=96)
    plt.close(fig)
