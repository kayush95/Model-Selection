import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

import pymc3 as pm
import theano.tensor as T
import theano

from scipy.stats import mode, chisquare

from sklearn.metrics import confusion_matrix, accuracy_score

import lasagne

import sys, os

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip
    # Check for data in folder, if not download
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)
    #Load labels
    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data
    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

input_var = theano.shared(X_train[:500, ...].astype(np.float64))
target_var = theano.shared(y_train[:500, ...].astype(np.float64))

def build_ann(init, input_var, target_var):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    with pm.Model() as neural_network:
        n_hid1 = 500
        l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=n_hid1,
            nonlinearity=lasagne.nonlinearities.tanh,
            b=init,
            W=init
        )
        n_hid2 = 500
        # Another 800-unit layer:
        l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=n_hid2,
            nonlinearity=lasagne.nonlinearities.tanh,
            b=init,
            W=init
        )
        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
        l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax,
            b=init,
            W=init
        )
        prediction = lasagne.layers.get_output(l_out)
        # 10 discrete output classes -> pymc3 categorical distribution
        out = pm.Categorical('out', 
                            prediction,
                            observed=target_var)
    return neural_network

class GaussWeights(object):
    def __init__(self):
        self.count = 0
    def __call__(self, shape):
        self.count += 1
        return pm.Normal('w%d' % self.count, mu=0, sd=.1, 
                         testval=np.random.normal(size=shape).astype(np.float64),
                         shape=shape)

class GaussWeightsHierarchicalRegularization(object):
    def __init__(self):
        self.count = 0
    def __call__(self, shape):
        self.count += 1
        
        regularization = pm.HalfNormal('reg_hyper%d' % self.count, sd=1)
        
        return pm.Normal('w%d' % self.count, mu=0, sd=regularization, 
                         testval=np.random.normal(size=shape),
                         shape=shape)


minibatch_x = pm.Minibatch(X_train, batch_size=500, dtype='float64')
minibatch_y = pm.Minibatch(y_train, batch_size=500, dtype='float64')

#neural_network = build_ann_keras(init = GaussWeights_keras(), x = minibatch_x, y = minibatch_y)
neural_network = build_ann(init = GaussWeightsHierarchicalRegularization(), input_var = input_var, target_var = target_var)

with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(50000, more_replacements={input_var: minibatch_x, target_var:minibatch_y}, method=inference)

trace = approx.sample(draws = 500)
trace = trace[200:]
input_var.set_value(X_test)
target_var.set_value(y_test)

with neural_network:
    ppc = pm.sample_ppc(trace, samples=100)

y_pred = mode(ppc['out'], axis=0).mode[0, :]

print('Accuracy on test data for FNN = {}%'.format(accuracy_score(y_test, y_pred) * 100))

waic = pm.waic(trace, neural_network)
loo = pm.loo(trace, neural_network)

print("WAIC for FNN network: ", {waic.WAIC})
print("LOO for FNN network: ", {loo.LOO})

def build_ann_conv(init, input_var, target_var):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.tanh,
            W=init)

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.tanh,
        W=init)
    
    network = lasagne.layers.MaxPool2DLayer(network, 
                                            pool_size=(2, 2))
    
    n_hid2 = 256
    network = lasagne.layers.DenseLayer(
        network, num_units=n_hid2,
        nonlinearity=lasagne.nonlinearities.tanh,
        b=init,
        W=init
    )

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    network = lasagne.layers.DenseLayer(
        network, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax,
        b=init,
        W=init
    )
    
    prediction = lasagne.layers.get_output(network)
    
    return pm.Categorical('out', 
                   prediction,
                   observed=target_var)

input_var_conv = theano.shared(X_train[:500, ...].astype(np.float64))
target_var_conv = theano.shared(y_train[:500, ...].astype(np.float64))

conv_neural_network = build_ann_conv(init = GaussWeights(), input_var = input_var_conv, target_var = target_var_conv)

with conv_neural_network:
    inference = pm.ADVI()
    approx_conv = pm.fit(50000, more_replacements={input_var: minibatch_x, target_var:minibatch_y}, method=inference)

trace_conv = approx.sample(draws = 500)
trace_conv[:200]
input_var_conv.set_value(X_test)
target_var_conv.set_value(y_test)

with conv_neural_network:
    ppc_conv = pm.sample_ppc(trace_conv, samples=100)

y_pred_conv = mode(ppc_conv['out'], axis=0).mode[0, :]

print('Accuracy on test data for CNN = {}%'.format(accuracy_score(y_test, y_pred_conv) * 100))

waic_conv = pm.waic(trace_conv, conv_neural_network)
loo_conv = pm.loo(trace_conv, conv_neural_network)

print("WAIC for CNN network: ", {waic_conv.WAIC})
print("LOO for CNN network: ", {loo_conv.LOO})

