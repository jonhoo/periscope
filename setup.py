#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='Periscope',
    version='0.0.1',
    description='6.869 final project - Theano network for MiniPlaces',
    long_description=open('README.md', 'r').read(),
    author='Jon Gjengset',
    author_email='jon@thesquareplanet.com',
    url='https://github.com/jonhoo/periscope',
    packages=[],
    install_requires=[
        'Lasagne == 0.2.dev1',
        'numpy',
        'pillow',
        'progressbar2',
        'scipy',
        'termcolor',
        'Theano >= 0.7.0'
    ],
    dependency_links=[
        'https://github.com/Lasagne/Lasagne/archive/master.zip#egg=Lasagne-0.2.dev1',
        'https://github.com/Theano/Theano/archive/master.zip#egg=Theano-0.7.0'
    ],
    scripts=['main.py'],
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Education"
    ]
)
