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
# for some reason pip is not following dependency_links to get Lasagne 0.2.dev1
python3 -m pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
# pip install -e works for everything else.
python3 -m pip install --upgrade -e .
