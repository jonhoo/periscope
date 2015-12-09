Periscope is a working solution to the [MIT
6.869](http://6.869.csail.mit.edu/) [Mini Places
Challenge](http://6.869.csail.mit.edu/fa15/project.html) (a subset of
the [Places2 Challenge](http://places2.csail.mit.edu/)) built using the
deep-learning framework [Lasagne](https://github.com/Lasagne/Lasagne).
At the time of writing, this network achieves a
[test-set](http://places2.csail.mit.edu/) top-5 accuracy of 77.5%, and
an exact match accuracy of 47.9%; one of the best of the 2015 Fall
semester contenders.

## Approach

In order to match the performance of the [6.869 matconvnet reference
network](http://6.869.csail.mit.edu/fa15/challenge/miniplacesCNN.zip) in
our Lasagne-based implementation, we found that input augmentation was
key. The table below shows the top-5 classification error of one version
of our network with different augmentation strategies used to increase
the training set. As evidenced from the results in the table,
augmentation can yield significantly improved classification accuracy,
and can substantially reduce overfitting.

| Strategy        | Validation | Training
| --------------- | ---------: | -------:
| No augmentation |        36% |       4%
| Only cropping   |        32% |      12%
| Only flipping   |        30% |      15%
| Both            |        29% |      19%

Overfitting is a substantial problems for neural networks, especially as
they grow even deeper. In particular, the issue arises because the
network has many more free variables than the number of images in the
training set, which lets it learn features that are specific to the
training set images, essentially memorizing the images, instead of
building generalized feature detectors. Input augmentation is designed
to help prevent this by artificially expanding the training data such
that the network no longer has sufficient capacity to memorize the
training set, and so is forced to generalize.

To facilitate easy experimentation with augmentation techniques, we
implemented our system such that augmentation is done on-line during
training, rather than as a pre-processing step. This gives us feedback
on how well different augmentation strategies work quickly, and does not
requires us to re-process the entire dataset in order to try a new
approach.  Furthermore, it gives us greater flexibility in designing new
augmentation techniques, as we do not have to worry about fitting a much
larger image set in memory or on disk for performance.

In Periscope, we implement cropping and flipping using symbolic Theano
expressions, which allows us to take advantage of the parallelism
provided by the GPU to speed up the augmentation significantly. We
further speed up the training process by applying the same
transformation to all images in each batch, and only varying the
transformation parameters between batches. During evaluation, we apply
flipping and multiple crops to each image, and categorize the image
according to the median confidence of each category across the different
evaluations. This last technique alone gives us a several
percentage-point accuracy improvement.

Finally, we apply multiple different trained models to the test set, and
compute the median probability of each category for each image. We then
rank the top five categories based on this median value. Combining
multiple models in this way yields another two percentage-points in
overall accuracy.

## Running

To run the network, change the paths in the `Makefile` to match where
you have downloaded the [image
data](http://6.869.csail.mit.edu/fa15/challenge/data.tar.gz) and the
[development
kit](http://6.869.csail.mit.edu/fa15/challenge/development_kit.tar.gz).
Then run `make solve` to start training. You can switch between
different network layouts using `make NET=smarter solve`.

By default, the trained network parameters will be stored in
`exp-$(NET)/epoch-*.mdl`. While training, a plot of loss and error rate
will also be produced in `exp-$(NET)/plot.png`. To do more specialized
plotting, or to plot multiple models on the same graph, use `plot.py`.

To evaluate the network, use `evaluate.py`. Note that you can pass
multiple trained models to `evaluate.py` using the `-m` flag. You will
need to pass a matching number of network names using `-n` to tell the
script what network layout each model was trained with. To achieve our
top performance, combine multiple `NET=smarter` trained models. If you
want textual labels for each image, use `-l`. The default output format
is catered to the format used by the [6.869
leaderboard](http://miniplaces.csail.mit.edu/leaderboard.php).

## Experimentation

A key component of deep learning systems that wish to achieve good
accuracy on limited datasets is input augmentation. By automatically
expanding the training set, the network is exposed to greater variation
in its inputs, reducing overfitting by teaching the network to
generalize.

Traditional input augmentation yields *positive examples*; additional
input images that have been perturbed in non-destructive ways so that
they remain similar to the images they were constructed from, and so
still fit the labels assigned to the original data. We propose also
augmenting the dataset with *negative examples*; additional input images
that teach the network what features are **not** important.

Furthermore, we apply recent techniques to detect which parts of
incorrectly labelled training images "distracted" the network. From
this, we construct two new training datasets: one that has these
distractions masked out, and one that has the images with only the
distractions present. We then train the network with the former as
positive examples, and the latter as negative examples. Our experimental
results show that these techniques do not improve overall accuracy
whatsoever, but we believe they are interesting nonetheless.

The implementation of these additional techniques can be found in the
various branches of this repository. In particular, the branches
`focused-retrain`, `mixin`, and `negative-examples` may be of interest.
