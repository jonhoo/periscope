#!/usr/bin/env python3

import argparse
import numpy
import sys
import re
import os
import os.path
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('model',
                    type=argparse.FileType('rb'),
                    nargs='+',
                    help='path to .mdl to extract plot data from'
                    )
parser.add_argument('-f', '--format', help='image format', default=None)
parser.add_argument('-n', '--names', help='set experiment names', nargs="+")
parser.add_argument('-t', '--title', help='plot title', default=None)
parser.add_argument('-s', '--set', help='plot only the given dataset', choices=['training', 'validation', 'all'], default='all')
parser.add_argument('-k', '--atk', help='plot only accuracy at top-k', type=int, choices=[5, 1, 0], default=0)
parser.add_argument('-m', '--max', help='y upper limit', type=float, default=1.0)
parser.add_argument('-c', '--column', help='make plot fit single-column', action='store_true', default=False)
args = parser.parse_args()

import matplotlib
if args.format is not None:
    matplotlib.use('Agg') # avoid the need for X
import seaborn as sns
fs = 1.0
if not args.column:
    fs *= 1.5
font = 'sans-serif'
if args.format == 'eps':
    fs *= 0.7
    font = 'serif'
sns.set(style="darkgrid", color_codes=True, font_scale=fs, font=font)
import matplotlib.pyplot as plt

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

if not args.column:
    fig = plt.figure()
else:
    fig = plt.figure(figsize=(3.25, 4))
ax_err = fig.gca()

# plot error
ax_err.set_xlim(1, maxe+1)
ax_err.set_ylim(0, args.max)
ax_err.yaxis.set_ticks(numpy.arange(0.0, args.max+0.1, 0.1))
if args.title is not None:
    ax_err.set_title(args.title)

ms = 4
if args.column:
    ms = 3

tlegends = []
for i in range(len(training)):
    if args.names and i < len(args.names):
        model = args.names[i]
    else:
        model = re.sub('\.mdl$', '', args.model[i].name)
        model = re.sub(os.path.sep + 'epoch-\d+$', '', model)
        if os.path.sep in model:
            model = os.path.basename(model)

    # exact is s, top5 is o
    # training is '--', validation is ''

    c = None
    xend = len(training[i])+1
    if args.atk != 5 and args.set != 'validation':
        c = ax_err.plot(range(1, xend), [1-dp[1] for dp in training[i]], '--', color=c, marker='s', markersize=ms)
        c = c[0].get_color()
        tlegends.append('{}, training, exact'.format(model))
    if args.atk != 1 and args.set != 'validation':
        c = ax_err.plot(range(1, xend), [1-dp[2] for dp in training[i]], '--', color=c, marker='o', markersize=ms)
        c = c[0].get_color()
        tlegends.append('{}, training, top 5'.format(model))

    xend = len(validation[i])+1
    if args.atk != 5 and args.set != 'training':
        c = ax_err.plot(range(1, xend), [1-dp[1] for dp in validation[i]], '', color=c, marker='s', markersize=ms)
        c = c[0].get_color()
        tlegends.append('{}, validation, exact'.format(model))
    if args.atk != 1 and args.set != 'training':
        c = ax_err.plot(range(1, xend), [1-dp[2] for dp in validation[i]], '', color=c, marker='o', markersize=ms)
        c = c[0].get_color()
        tlegends.append('{}, validation, top 5'.format(model))

if args.atk == 0 and args.set == 'all':
    ax_err.legend(tlegends, ncol=len(training))
else:
    ax_err.legend(tlegends)

if args.format is None:
    plt.show(fig)
else:
    fig.savefig(sys.stdout, format=args.format)
    plt.close(fig)
