# coding: utf-8
__author__ = 'fgu041513'

import numpy as np
from matplotlib import pyplot as plt
from BSpricer import *
from scipy.special import expit, logit
import pickle

filename = "BS_stat_data"

with open(filename, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    Xall, Yall, = u.load()

X = Xall[:10000]
Y = Yall[:10000]
Xvalid = Xall[10000:20000]
Yvalid = Yall[10000:20000]
    
# NEURAL NETWORK
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, Dropout
from keras.regularizers import WeightRegularizer

input_dim = X[0].shape[0]
output_dim = 1

W_regularizer = WeightRegularizer(l1=0., l2=0.)
model = Sequential()
model.add(Dense(10, input_dim=input_dim, activation='sigmoid', W_regularizer=W_regularizer))
model.add(Dense(10, activation='sigmoid', W_regularizer=W_regularizer))
model.add(Dense(10, activation='sigmoid', W_regularizer=W_regularizer))
model.add(Dense(output_dim, name='output_layer', activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='mse')
print(model.summary())

# fit
res = model.fit(X, Y, batch_size=50, nb_epoch=50, validation_data = (Xvalid, Yvalid))
plt.figure()
plt.plot(res.epoch, res.history['loss'])

#model_file_name = "BS/model.h5"
#model.save(model_file_name)
#model = load_model(model_file_name)

# predict
S = np.arange(0.1, 1.1, 0.01)
expiry = np.array([0.5] * S.shape[0])
strike = np.array([0.5] * S.shape[0])
sigma = np.array([0.2] * S.shape[0])
X = np.transpose(np.array([S, expiry, strike, sigma]))
Y = np.array([ModelBlackScholes(x[0], x[1], PayoffCall(x[2]), sigma=x[3])() / x[0] for x in X])

moneyness = strike / S
sigmat = sigma * np.array([m.sqrt(x) for x in expiry])
X = np.transpose(np.array([moneyness, sigmat]))
Yhat = model.predict(X)

plt.figure()
plt.plot(S, Y)
plt.plot(S, Yhat)

# Implied vol
S = np.arange(0.35, 0.81, 0.01)
expiry = np.array([0.5] * S.shape[0])
strike = np.array([0.5] * S.shape[0])
sigma = np.array([0.2] * S.shape[0])
X_BS = np.transpose(np.array([S, expiry, strike, sigma]))
Y = np.array([ModelBlackScholes(x[0], x[1], PayoffCall(x[2]), sigma=x[3])() / x[0] for x in X_BS])

moneyness = strike / S
sigmat = sigma * np.array([m.sqrt(x) for x in expiry])
X_NN = np.transpose(np.array([moneyness, sigmat]))
Yhat = model.predict(X_NN)
sigmahat = np.array([implied_sigma(x[0], x[1], PayoffCall, x[2], 0.0, p[0] * x[0]) for x, p in zip(X_BS, Yhat)])

    
plt.figure()
plt.plot(moneyness, sigma)
plt.plot(moneyness, sigmahat)
