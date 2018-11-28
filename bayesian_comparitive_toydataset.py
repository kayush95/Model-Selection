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
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons

floatX=theano.config.floatX

X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

#f1 = plt.figure(1)
#fig, ax = plt.subplots(figsize=(12, 8))
#ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
#ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
#sns.despine(); ax.legend()
#ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');

def construct_nn_1(ann_input, ann_output):
    n_hidden = 5
    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)
    with pm.Model() as neural_network:
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
                                 shape=(X.shape[1], n_hidden), 
                                 testval=init_1)
        # Weights from hidden layer to output
        weights_2_out = pm.Normal('w_2_out', 0, sd=1, 
                                  shape=(n_hidden,), 
                                  testval=init_out)
        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, 
                                         weights_in_1))
        act_out = pm.math.sigmoid(pm.math.dot(act_1, 
                                              weights_2_out))
        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out',
                           act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network


def construct_nn_2(ann_input, ann_output):
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

def construct_nn_3(ann_input, ann_output):
    n_hidden = 5
    
    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_3 = np.random.randn(n_hidden, n_hidden).astype(floatX)
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
        
        # Weights from 2nd to 3rd layer
        weights_2_3 = pm.Normal('w_2_3', 0, sd=1, 
                                shape=(n_hidden, n_hidden), 
                                testval=init_3)
        
        # Weights from hidden layer to output
        weights_3_out = pm.Normal('w_3_out', 0, sd=1, 
                                  shape=(n_hidden,), 
                                  testval=init_out)
        
        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, 
                                         weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, 
                                         weights_1_2))
        act_3 = pm.math.tanh(pm.math.dot(act_2, 
                                         weights_2_3))
        act_out = pm.math.sigmoid(pm.math.dot(act_3, 
                                              weights_3_out))
        
        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out', 
                           act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network

def construct_nn_4(ann_input, ann_output):
    n_hidden = 5
    
    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_3 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_4 = np.random.randn(n_hidden, n_hidden).astype(floatX)
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
        
        # Weights from 2nd to 3rd layer
        weights_2_3 = pm.Normal('w_2_3', 0, sd=1, 
                                shape=(n_hidden, n_hidden), 
                                testval=init_3)
        
        # Weights from 3rd to 4thd layer
        weights_3_4 = pm.Normal('w_3_4', 0, sd=1, 
                                shape=(n_hidden, n_hidden), 
                                testval=init_4)
        
        # Weights from hidden layer to output
        weights_4_out = pm.Normal('w_4_out', 0, sd=1, 
                                  shape=(n_hidden,), 
                                  testval=init_out)
        
        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, 
                                         weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, 
                                         weights_1_2))
        act_3 = pm.math.tanh(pm.math.dot(act_2, 
                                         weights_2_3))
        act_4 = pm.math.tanh(pm.math.dot(act_3, 
                                         weights_3_4))
        act_out = pm.math.sigmoid(pm.math.dot(act_3, 
                                              weights_4_out))
        
        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out', 
                           act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network

def construct_nn_5(ann_input, ann_output):
    n_hidden = 5
    
    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_3 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_4 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_5 = np.random.randn(n_hidden, n_hidden).astype(floatX)
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
        
        # Weights from 2nd to 3rd layer
        weights_2_3 = pm.Normal('w_2_3', 0, sd=1, 
                                shape=(n_hidden, n_hidden), 
                                testval=init_3)
        
        # Weights from 3rd to 4thd layer
        weights_3_4 = pm.Normal('w_3_4', 0, sd=1, 
                                shape=(n_hidden, n_hidden), 
                                testval=init_4)

        weights_4_5 = pm.Normal('w_4_5', 0, sd=1, 
                                shape=(n_hidden, n_hidden), 
                                testval=init_5)
        
        # Weights from hidden layer to output
        weights_5_out = pm.Normal('w_5_out', 0, sd=1, 
                                  shape=(n_hidden,), 
                                  testval=init_out)
        
        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, 
                                         weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, 
                                         weights_1_2))
        act_3 = pm.math.tanh(pm.math.dot(act_2, 
                                         weights_2_3))
        act_4 = pm.math.tanh(pm.math.dot(act_3, 
                                         weights_3_4))
        act_5 = pm.math.tanh(pm.math.dot(act_3, 
                                         weights_4_5))
        act_out = pm.math.sigmoid(pm.math.dot(act_3, 
                                              weights_5_out))
        
        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli('out', 
                           act_out,
                           observed=ann_output,
                           total_size=Y_train.shape[0] # IMPORTANT for minibatches
                          )
    return neural_network

ann_input = theano.shared(X_train)
ann_output = theano.shared(Y_train)
neural_network_1 = construct_nn_1(ann_input, ann_output)
neural_network_2 = construct_nn_2(ann_input, ann_output) 
neural_network_3 = construct_nn_3(ann_input, ann_output) 
neural_network_4 = construct_nn_4(ann_input, ann_output) 
neural_network_5 = construct_nn_5(ann_input, ann_output) 

from pymc3.theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))

with neural_network_1:
    inference = pm.ADVI()
    approx1 = pm.fit(n=50000, method=inference)
    trace_1 = approx1.sample(draws = 5000)

with neural_network_2:
    inference = pm.ADVI()
    approx2 = pm.fit(n=50000, method=inference)
    trace_2 = approx2.sample(draws = 5000)

with neural_network_3:
    inference = pm.ADVI()
    approx3 = pm.fit(n=50000, method=inference)
    trace_3 = approx3.sample(draws = 5000)

with neural_network_4:
    inference = pm.ADVI()
    approx4 = pm.fit(n=50000, method=inference)
    trace_4 = approx4.sample(draws = 5000)

with neural_network_5:
    inference = pm.ADVI()
    approx5 = pm.fit(n=50000, method=inference)
    trace_5 = approx5.sample(draws = 5000)

ann_input.set_value(X_test)
ann_output.set_value(Y_test)

with neural_network_1:
    ppc1 = pm.sample_ppc(trace_1[1000:], samples=2000, progressbar=False)

pred1 = ppc1['out'].mean(axis=0) > 0.5

print(f'Accuracy for 1st network= {(Y_test == pred1).mean() * 100}%')

with neural_network_2:
    ppc2 = pm.sample_ppc(trace_2[1000:], samples=2000, progressbar=False)

pred2 = ppc2['out'].mean(axis=0) > 0.5

print(f'Accuracy for 2nd network= {(Y_test == pred2).mean() * 100}%')

with neural_network_3:
    ppc3 = pm.sample_ppc(trace_3[1000:], samples=2000, progressbar=False)

pred3 = ppc3['out'].mean(axis=0) > 0.5

print(f'Accuracy for 3rd network= {(Y_test == pred3).mean() * 100}%')

with neural_network_4:
    ppc4 = pm.sample_ppc(trace_4[1000:], samples=2000, progressbar=False)

pred4 = ppc4['out'].mean(axis=0) > 0.5

print(f'Accuracy for 4th network= {(Y_test == pred4).mean() * 100}%')

with neural_network_5:
    ppc5 = pm.sample_ppc(trace_5[1000:], samples=2000, progressbar=False)

pred5 = ppc5['out'].mean(axis=0) > 0.5

print(f'Accuracy for 5th network= {(Y_test == pred5).mean() * 100}%')

waic_1 = pm.waic(trace_1[1000:], neural_network_1)
loo_1 = pm.loo(trace_1[1000:], neural_network_1)

print(f"WAIC for 1st network: {waic_1.WAIC}")
print(f"LOO for 1st network: {loo_1.LOO}")

waic_2 = pm.waic(trace_2[1000:], neural_network_2)
loo_2 = pm.loo(trace_2[1000:], neural_network_2)

print(f"WAIC for 2nd network: {waic_2.WAIC}")
print(f"LOO for 2nd network: {loo_2.LOO}")

waic_3 = pm.waic(trace_3[1000:], neural_network_3)
loo_3 = pm.loo(trace_3[1000:], neural_network_3)

print(f"WAIC for 3rd network: {waic_3.WAIC}")
print(f"LOO for 3rd network: {loo_3.LOO}")

waic_4 = pm.waic(trace_4[1000:], neural_network_4)
loo_4 = pm.loo(trace_4[1000:], neural_network_4)

print(f"WAIC for 4th network: {waic_4.WAIC}")
print(f"LOO for 4th network: {loo_4.LOO}")

waic_5 = pm.waic(trace_5[1000:], neural_network_5)
loo_5 = pm.loo(trace_5[1000:], neural_network_5)

print(f"WAIC for 5th network: {waic_5.WAIC}")
print(f"LOO for 5th network: {loo_5.LOO}")
f2 = plt.figure(1)
plt.plot((1,2,3,4,5), (waic_1,waic_2,waic_3,waic_4,waic_5),'g',(1,2,3,4,5),(loo_1,loo_2,loo_3,loo_4,loo_5),'b')

f2.savefig('waic-loo_trend.jpg')

#f3 = plt.figure(3)
#pm.traceplot(trace_1);
#pm.traceplot(trace_2);
#pm.traceplot(trace_3);
#pm.traceplot(trace_4);
#pm.traceplot(trace_5);