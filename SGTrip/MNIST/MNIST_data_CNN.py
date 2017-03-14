# coding: utf-8
__author__ = 'fgu041513'

import pickle
import gzip
import numpy as np
from matplotlib import pyplot as plt

# LOADING
filename="mnist.pkl.gz"

with gzip.open(filename, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()

print(training_data[0].__len__())
print(validation_data[0].__len__())
print(test_data[0].__len__())


# NEURAL NETWORK
# Convert Y to dummy
from keras.utils import np_utils
# from keras.utils.visualize_util import plot as keras_plot

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import WeightRegularizer

data = test_data#training_data
n = 28
X = (data[0]).reshape(data[0].shape[0], 1, n, n)
Y = np_utils.to_categorical(data[1])
Xvalid = (validation_data[0]).reshape(validation_data[0].shape[0], 1, n, n)
Yvalid = np_utils.to_categorical(validation_data[1])
input_dim = X[0].shape
output_dim = 10
W_regularizer = WeightRegularizer(l1=0., l2=0.)

model = Sequential()
model.add(Convolution2D(20, 5, 5, input_shape=(1, n, n), activation='relu', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(Flatten())
model.add(Dense(100, name='hidden_layer', activation='sigmoid', W_regularizer=W_regularizer))
model.add(Dense(output_dim, name='output_layer', activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
# keras_plot(model, to_file='model.png')
# fit
res = model.fit(X, Y, batch_size=10, nb_epoch=15, validation_data=(Xvalid, Yvalid))
plt.figure()
plt.plot(res.epoch, res.history['loss'])

plt.figure()
plt.plot(res.epoch, res.history['acc'])
plt.plot(res.epoch, res.history['val_acc'])

# predict
data = training_data#test_data
X = (data[0]).reshape(data[0].shape[0], 1, n, n)
Y = data[1]
Yhat = model.predict(X)

print(np.mean(np.argmax(Yhat, axis=1) == Y))
