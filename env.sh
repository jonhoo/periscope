#!/bin/sh
# Set up the environment needed for our build.  Tries to keep
# everything except pip3, python3, and virtualenv inside virutalenv.

set -e
command -v python3 >/dev/null 2>&1 || { \
  echo >&2 "python3 is required"; sudo apt-get install python3; }
command -v pip3 >/dev/null 2>&1 || { \
  echo >&2 "pip3 is required"; sudo apt-get install python3-pip; }
python3 -c "import virtualenv" >/dev/null 2>&1 || { \
  echo >&2 "virtualenv is required"; sudo pip3 install virtualenv; }

if [ ! -x env/bin/python3 ] || [ ! -f env/bin/activate ]; then
  rm -rf env
  python3 -m virtualenv env
fi
. env/bin/activate

# numpy isn't listed as a dependency in scipy, so we need to do it by hand
python3 -m pip wheel numpy

# Use Theano from the latest on github.
if [ ! -d env/src/theano/.git ]; then
  git clone https://github.com/Theano/Theano.git env/src/theano
fi

# Update to latest; discard any local changes.
git -C env/src/theano fetch --multiple origin
git -C env/src/theano checkout master
git -C env/src/theano reset --hard origin/master
python3 -m pip install --upgrade env/src/theano

# Also use Lasagne from the latest on github, but also patch in batchnorm.
if [ ! -d env/src/lasagne/.git ]; then
  git clone https://github.com/Lasagne/Lasagne.git env/src/lasagne
  git -C env/src/lasagne remote add f0k https://github.com/f0k/Lasagne.git
fi

# Get latest lasagne and apply batcnnorm patch into local master branch.
git -C env/src/lasagne fetch --multiple origin f0k
git -C env/src/lasagne checkout master
git -C env/src/lasagne reset --hard origin/master
git -C env/src/lasagne merge -s ours f0k/batchnorm -m "Patched batchnorm."
python3 -m pip install --upgrade env/src/lasagne

# pip install -e works for everything else.
python3 -m pip install --upgrade -e .
# update timestamp
touch env/bin/python3
touch env
# exit the venv
deactivate
