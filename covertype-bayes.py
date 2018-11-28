import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import numpy as np
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pymc3.backends import HDF5, text
from scipy.stats import mode, chisquare

from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('covtype_preprocess.csv', index_col=0)

df['Cover_Type'] = df['Cover_Type'].apply(lambda x: str(x))

output_col = 'Cover_Type'
input_cols = [c for c in df.columns if c != output_col]
input_formula = ''.join(c + ' + ' for c in input_cols)
input_formula = input_formula + '-1'

import patsy
from sklearn.preprocessing import scale, normalize
X = patsy.dmatrix(formula_like=input_formula, 
                  data=df, 
                  return_type='dataframe')
# X = normalize(X)

Y = patsy.dmatrix(formula_like='Cover_Type -1',
                  data=df,
                  return_type='dataframe')
print(X.shape, Y.shape)

def make_nn(ann_input, ann_output, n_hidden):
    """
    Makes a feed forward neural network with n_hidden layers for doing multi-
    class classification.
    Feed-forward networks are easy to define, so I have not relied on any 
    other Deep Learning frameworks to define the neural network here.
    """
    init_1 = np.random.randn(ann_input.shape[1], n_hidden)
    init_2 = np.random.randn(n_hidden, n_hidden)
    init_out = np.random.randn(n_hidden, ann_output.shape[1])
    with pm.Model() as nn_model:
        # Define weights
        weights_1 = pm.Normal('w_1', mu=0, sd=1, 
                              shape=(ann_input.shape[1], n_hidden),
                              testval=init_1)
        weights_2 = pm.Normal('w_2', mu=0, sd=1,
                              shape=(n_hidden, n_hidden),
                              testval=init_2)
        weights_out = pm.Normal('w_out', mu=0, sd=1, 
                                shape=(n_hidden, ann_output.shape[1]),
                                testval=init_out)
        # Define activations
        acts_1 = pm.Deterministic('activations_1', 
                                  tt.tanh(tt.dot(ann_input, weights_1)))
        acts_2 = pm.Deterministic('activations_2', 
                                  tt.tanh(tt.dot(acts_1, weights_2)))
        acts_out = pm.Deterministic('activations_out', 
                                    tt.nnet.softmax(tt.dot(acts_2, weights_out)))  # noqa
        # Define likelihood
        out = pm.Multinomial('likelihood', n=1, p=acts_out, 
                             observed=ann_output)
    return nn_model

from sklearn.preprocessing import MinMaxScaler

downsampled_targets = []

for i in range(1, 7+1):
    # print(f'target[{i}]')
    target = Y[Y['Cover_Type[{i}]'.format(i=i)] == 1]
    # print(len(target))
    downsampled_targets.append(target.sample(2747))


mms = MinMaxScaler()
X_tfm = pm.floatX(mms.fit_transform(X[input_cols]))

Y_downsampled = pd.concat(downsampled_targets)
Y_downsamp = pm.floatX(Y_downsampled)

X_downsampled = X_tfm[Y_downsampled.index]
X_downsamp = pm.floatX(X_downsampled)
n_hidden = 20
model = make_nn(X_downsamp, Y_downsamp, n_hidden=n_hidden)
with model:
    s = theano.shared(pm.floatX(1.1))
    inference = pm.ADVI(cost_part_grad_scale=s)
    approx = pm.fit(200000, method=inference,
                    callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-1)])

plt.plot(approx.hist)
plt.yscale('log')

with model:
    trace = approx.sample(1000)

with model:
    samp_ppc = pm.sample_ppc(trace[100:], samples=1000)

preds_proba = samp_ppc['likelihood'].mean(axis=0)
preds = (preds_proba == np.max(preds_proba, axis=1, keepdims=True)) * 1

from sklearn.metrics import classification_report
print(classification_report(Y_downsamp, preds))

waic = pm.waic(trace[100:], model)
loo = pm.loo(trace[100:], model)

print(f"WAIC for network: {waic.WAIC}")
print(f"LOO for network: {loo.LOO}")