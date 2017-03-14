from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import copy

import numpy as np
import pandas as pd

seed = 123
np.random.seed(seed)

data = pd.read_excel("data.xlsx", index_col=0)
print(data.columns)
plt.figure()
plt.plot(data['S&P 500'])
plt.show()
ret = data.pct_change()
ret = ret.ix[1:, :]

#ret = expit(ret)
ret.to_excel("output/actual.xlsx")

# this is the size of our encoded representations
encoding_dim = 2

nr, nc = ret.shape

# this is our input placeholder
input_data = Input(shape=(nc,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='linear')(input_data)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(nc, activation='linear')(encoded)


# this model maps an input to its reconstruction
autoencoder = Model(input=input_data, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_data, output=encoded)


# create a placeholder for an encoded (m-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


autoencoder.compile(optimizer='rmsprop', loss='mse')


x_train = np.array(ret)
res = autoencoder.fit(x_train, x_train, nb_epoch=100, batch_size=50)

plt.figure()
plt.plot(res.epoch, res.history['loss'])
plt.show()

encoded_series = encoder.predict(x_train)
decoded_series = decoder.predict(encoded_series)

encoded_series_pd = pd.DataFrame(encoded_series)
#encoded_series_pd = logit(encoded_series_pd)
encoded_series_pd.to_excel("output/res.xlsx")

decoded_series_pd = pd.DataFrame(decoded_series, columns=ret.columns)
decoded_series_pd.to_excel("output/decoded.xlsx")


# Check results
old_mean = decoded_series_pd.mean(axis=0)
new_mean = ret.mean(axis=0)
new_ret = decoded_series_pd.apply(lambda x: 1 + x - old_mean + new_mean, axis=1)

decoded_data = copy.deepcopy(data)
decoded_data.ix[1:] = new_ret.values
decoded_data = decoded_data.cumprod(axis=0)

col = data.columns[0]
plt.figure()
plt.plot(data[col])
plt.plot(decoded_data[col])
plt.show()

# Initial correlation matrix
plt.figure()
plt.matshow(data.corr())
x.set_xticklabels([''] + alpha)
# Sparse correlation matrix
plt.figure()
plt.matshow(decoded_series_pd.corr())
plt.show()

print("done")
