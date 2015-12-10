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
parser.add_argument('-n', '--names', help='set experiment names', nargs="+")
parser.add_argument('-s', '--set', help='plot only the given dataset', choices=['training', 'validation'], default='validation')
parser.add_argument('-k', '--atk', help='plot only accuracy at top-k', type=int, choices=[5, 1], default=1)
args = parser.parse_args()

if args.names is None:
    args.names = [m.name for m in args.model]

i = -1
for model in args.model:
    i += 1
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
        if args.set == "training":
            vals = pickle.load(model)
            pickle.load(model) # validation
        else:
            pickle.load(model) # training
            vals = pickle.load(model)

        if args.atk == 1:
            val = numpy.max([v[1] for v in vals])
        else:
            val = numpy.max([v[2] for v in vals])

        num = re.sub(r'^.*?(\d+(\.\d+)?).*$', r'\1', args.names[i])
        print("{}\t{}\t{}\t{}".format(args.names[i], epoch, val, num))
    except EOFError:
        print("Model {} is invalid".format(model.name))
        sys.exit(1)
