import lasagne
import theano
from lasagne.init import HeUniform
from lasagne.layers import DropoutLayer, FeaturePoolLayer, prelu
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import rectify, softmax
Conv2DLayer = lasagne.layers.Conv2DLayer
MaxPool2DLayer = lasagne.layers.MaxPool2DLayer
if theano.config.device.startswith("gpu"):
    import lasagne.layers.dnn
    # Force GPU implementations if a GPU is available.
    # Do not know why Theano is not selecting these impls
    # by default as advertised.
    if theano.sandbox.cuda.dnn.dnn_available():
        Conv2DLayer = lasagne.layers.dnn.Conv2DDNNLayer
        MaxPool2DLayer = lasagne.layers.dnn.MaxPool2DDNNLayer

def base(network, cropsz, batchsz):
    # 1st
    network = Conv2DLayer(network, 64, (8, 8), stride=2, nonlinearity=rectify)
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 2nd
    network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 3rd
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 4th
    network = lasagne.layers.DenseLayer(network, 512)
    network = BatchNormLayer(network, nonlinearity=rectify)

    return network

# This one achieves about 27.5% err@5
def deeper(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    network = Conv2DLayer(network, 64, (7, 7), stride=1)
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    network = Conv2DLayer(network, 112, (5, 5), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same')
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 5
    network = Conv2DLayer(network, 320, (3, 3), stride=1)
    network = BatchNormLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), stride=1)
    # network = DropoutLayer(network)
    network = BatchNormLayer(network, nonlinearity=rectify)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512)
    network = DropoutLayer(network)
    # network = BatchNormLayer(network, nonlinearity=rectify)

    return network


def slim(network, cropsz, batchsz):
    # 1st
    network = Conv2DLayer(network, 64, (5, 5), stride=2, W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 2nd
    network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 3rd
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 4th
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 5th
    network = lasagne.layers.DenseLayer(network, 512, W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)

    return network

def cslim(network, cropsz, batchsz):
    # 1st
    network = Conv2DLayer(network, 64, (5, 5), stride=2, W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 2nd
    network = Conv2DLayer(network, 96, (5, 5), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (5, 5), stride=2)
    # 3rd
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 4th
    network = Conv2DLayer(network, 128, (3, 3), stride=1, pad='same', W=HeUniform('relu'))
    network = prelu(network)
    network = DropoutLayer(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)
    # 5th
    network = lasagne.layers.DenseLayer(network, 512, nonlinearity=None)
    network = DropoutLayer(network)
    network = FeaturePoolLayer(network, 2)

    return network

def smarter(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    network = Conv2DLayer(network, 64, (7, 7), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    network = Conv2DLayer(network, 112, (5, 5), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 5
    network = Conv2DLayer(network, 320, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512,
        W=HeUniform('relu'))
    network = prelu(network)
    network = DropoutLayer(network)
    # network = BatchNormLayer(network, nonlinearity=rectify)

    return network

def choosy(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    network = Conv2DLayer(network, 64, (7, 7), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    network = Conv2DLayer(network, 112, (5, 5), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 3rd.  Data size 27 -> 13
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 11 -> 5
    network = Conv2DLayer(network, 320, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 5th. Data size 5 -> 3
    network = Conv2DLayer(network, 512, (3, 3), nonlinearity=None)
    network = prelu(network)
    network = BatchNormLayer(network)

    # 6th. Data size 3 -> 1
    network = lasagne.layers.DenseLayer(network, 512, nonlinearity=None)
    network = DropoutLayer(network)
    network = FeaturePoolLayer(network, 2)

    return network

def msra_a(network, cropsz, batchsz):
    network = BatchNormLayer(network)
    network = prelu(Conv2DLayer(network, 256, (3,3), stride=2, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 256, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 256, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 256, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 256, (3,3), stride=1, W=HeUniform('relu')))
    network = MaxPool2DLayer(network, (2,2), stride=2)

    network = BatchNormLayer(network)
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))

    network = BatchNormLayer(network)
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = prelu(Conv2DLayer(network, 512, (3,3), stride=1, W=HeUniform('relu')))
    network = MaxPool2DLayer(network, (2,2), stride=2) # technically a Spatial Pyramid Pooling

    network = BatchNormLayer(network)
    network = prelu(lasagne.layers.DenseLayer(network, 4096))
    network = DropoutLayer(network)
    network = prelu(lasagne.layers.DenseLayer(network, 4096))
    network = DropoutLayer(network)
    return network
