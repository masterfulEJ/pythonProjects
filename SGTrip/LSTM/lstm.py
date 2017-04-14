import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix
def create_data(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pd.read_excel('us_gdp.xlsx', index_col=0)
plt.figure()
plt.plot(dataframe)
dataset = dataframe.values

look_back = 5
X, Y = create_data(dataset, look_back)
lasta = dataset[len(dataset)-look_back:len(dataset), 0]
X = numpy.vstack([X, numpy.reshape(lasta, (1, lasta.shape[0]))])

# split into train and test sets
train_size = int(len(Y) * 0.67)
test_size = len(Y) - train_size
trainX, testX, testX1 = X[:train_size, :], X[train_size:len(Y), :], X[train_size:(len(Y)+1), :]
trainY, testY = Y[:train_size], Y[train_size:len(Y)]


# create and fit the LSTM network
model = Sequential()
#model.add(Dense(2, input_dim=look_back))
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
testX1 = numpy.reshape(testX1, (testX1.shape[0], 1, testX1.shape[1]))
model.add(LSTM(2, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
print(model.summary())
res = model.fit(trainX, trainY, nb_epoch=100, batch_size=10)

plt.figure()
plt.plot(res.epoch, res.history['loss'])
plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
testPredict1 = model.predict(testX1)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[(look_back-1):len(trainPredict)+(look_back-1), :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[(len(trainPredict)+(look_back-1)):, :] = testPredict1

# plot baseline and predictions
dataframe['train'] = pd.DataFrame(trainPredictPlot, index=dataframe.index)[0]
dataframe['test'] = pd.DataFrame(testPredictPlot, index=dataframe.index)[0]
dataframe.plot()
plt.show()

plt.figure()
plt.plot(dataframe['train'] - dataframe['US GDP YoY'])
plt.plot(dataframe['test'] - dataframe['US GDP YoY'])
