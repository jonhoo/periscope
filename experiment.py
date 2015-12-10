import lasagne
import theano
from lasagne.init import HeUniform
from lasagne.layers import DropoutLayer, FeaturePoolLayer, ConcatLayer, prelu
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

def gooey_gadget(network_in, conv_in, conv_out, stride):
    network_c = Conv2DLayer(network_in, conv_in, (1, 1),
        W=HeUniform('relu'))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_c = Conv2DLayer(network_c, conv_out, (3, 3), stride=stride,
        W=HeUniform('relu'))
    network_c = prelu(network_c)
    network_c = BatchNormLayer(network_c)
    network_p = MaxPool2DLayer(network_in, (3, 3), stride=stride)
    # network_p = Conv2DLayer(network_p, pool_out, (1, 1),
    #    W=HeUniform('relu'))
    # network_p = prelu(network_p)
    # network_p = BatchNormLayer(network_p)
    return ConcatLayer((network_c, network_p))

def gooey(network, cropsz, batchsz):
    # 1st. Data size 117 -> 111 -> 55
    # 117*117*32 = 438048
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 115*115*32 = 423200
    network = Conv2DLayer(network, 32, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 55*55*48 = 121000
    network = Conv2DLayer(network, 40, (3, 3), stride=1,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 2nd. Data size 55 -> 27
    # 27*27*48 = 67068
    network = Conv2DLayer(network, 92, (3, 3), stride=2,
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    # 27*27*64 = 46656
    network = Conv2DLayer(network, 64, (3, 3), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)

    # 3rd.  Data size 27 -> 13, 192 + 144
    # 13*13*192 = 32448
    network = Conv2DLayer(network, 192, (3, 3), stride=1, pad='same',
        W=HeUniform('relu'))
    network = prelu(network)
    network = BatchNormLayer(network)
    network = MaxPool2DLayer(network, (3, 3), stride=2)

    # 4th.  Data size 13 -> 11 -> 5
    # 11*11*320 = 38720
    network = gooey_gadget(network, 64, 128, 1) # 192 + 128 = 320 channels
    # 5*5*512 = 12800
    network = gooey_gadget(network, 64, 192, 2) # 320 + 192 = 512 channels

    # 5th. Data size 5 -> 3
    # 3*3*832 = 7488
    network = gooey_gadget(network, 80, 320, 1) # 512 + 320 = 832 channels

    # 6th. Data size 3 -> 1, 592 + 512 channels
    # 1*1*1344 = 1344
    network = gooey_gadget(network, 128, 512, 1) # 832 + 512 = 1344 channels

    return network

