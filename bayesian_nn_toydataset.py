import theano
import pymc3 as pm
import sklearn
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
import theano.tensor as T
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons

floatX=theano.config.floatX

X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
sns.despine(); ax.legend()
ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');


def construct_nn(ann_input, ann_output):
    n_hidden = 5
    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)
    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
                                 shape=(X.shape[1], n_hidden), 
                                 testval=init_1)
        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, sd=1, 
                                shape=(n_hidden, n_hidden), 
                                testval=init_2)
        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=1, 
                                  shape=(n_hidden,), 
                                  testval=init_out)
        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, 
                                         weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, 
                                         weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, 
                                              weights_2_out))
        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out', 
                           act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network

ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)
neural_network = construct_nn(ann_input, ann_output)

from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))

with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(n=50000, method=inference)

trace_posttrain = approx.sample(draws = 2000)

print(f"Trace summary after training is {pm.summary(trace_posttrain)}")

plt.plot(-inference.hist)
plt.ylabel('ELBO')
plt.xlabel('iteration');

ann_input.set_value(X_test)
ann_output.set_value(Y_test)

with neural_network:
    ppc = pm.sample_ppc(trace_posttrain, samples=500, progressbar=False)

# Use probability of > 0.5 to assume prediction of class 1
pred = ppc['out'].mean(axis=0) > 0.5

fig, ax = plt.subplots()
ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
sns.despine()
ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y');

print(f'Accuracy = {(Y_test == pred).mean() * 100}%')

posttrain_waic = pm.waic(trace_posttrain, neural_network)
posttrain_loo = pm.loo(trace_posttrain, neural_network)

print(f"WAIC after training: {posttrain_waic.WAIC}")
print(f"LOO after training: {posttrain_loo.LOO}")

minibatch_x = pm.Minibatch(X_train, batch_size=32)
minibatch_y = pm.Minibatch(Y_train, batch_size=32)

ann_input_init = theano.shared(X_train)
ann_output_init = theano.shared(Y_train)
neural_network_minibatch = construct_nn(ann_input_init, ann_output_init)

with neural_network_minibatch:
    inference = pm.ADVI()
    approx_mini = pm.fit(40000, more_replacements={ann_input_init: minibatch_x, ann_output_init:minibatch_y}, method=inference)

trace_mini = approx_mini.sample(draws = 2000)

ann_input_init.set_value(X_test)
ann_output_init.set_value(Y_test)

with neural_network_minibatch:
    ppc_mini = pm.sample_ppc(trace_mini, samples=500, progressbar=False)

pred_mini = ppc_mini['out'].mean(axis=0) > 0.5
print(f'Accuracy for minibatch= {(Y_test == pred_mini).mean() * 100}%')

mini_waic = pm.waic(trace_mini, neural_network_minibatch)
mini_loo = pm.loo(trace_mini, neural_network_minibatch)

print(f"WAIC after training: {mini_waic.WAIC}")
print(f"LOO after training: {mini_loo.LOO}")