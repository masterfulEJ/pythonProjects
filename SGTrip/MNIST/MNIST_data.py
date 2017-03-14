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


# VISUALIZE DATA
i = 0
example = training_data[0][i]
example = np.reshape(example, (28, 28))
plt.imshow(example, interpolation='nearest', cmap='Greys')
plt.title(training_data[1][i])
plt.show()
i += 1


# NEURAL NETWORK
# Convert Y to dummy
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense, Convolution2D
from keras.regularizers import WeightRegularizer

data = training_data
X = data[0]
Y = np_utils.to_categorical(data[1])
Xvalid = validation_data[0]
Yvalid = np_utils.to_categorical(validation_data[1])
input_dim = X[0].shape[0]
output_dim = 10

W_regularizer = WeightRegularizer(l1=0., l2=0.)
model = Sequential()
model.add(Dense(30, input_dim=input_dim, name='hidden_layer', activation='sigmoid', W_regularizer=W_regularizer))
model.add(Dense(output_dim, name='output_layer', activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# fit
res = model.fit(X, Y, batch_size=10, nb_epoch=10, validation_data=(Xvalid, Yvalid))
plt.figure()
plt.plot(res.epoch, res.history['loss'])

plt.figure()
plt.plot(res.epoch, res.history['acc'])
plt.plot(res.epoch, res.history['val_acc'])

# predict
data = test_data
X = data[0]
Y = data[1]
Yhat = model.predict(X)

print(np.mean(np.argmax(Yhat, axis=1) == Y))

