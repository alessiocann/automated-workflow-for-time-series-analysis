from datetime import datetime, timedelta
import pandas as pd
import numpy
from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return asarray(X), asarray(y)





series = pd.read_csv('nuovo_ping.csv', header=0, parse_dates=[0], index_col=[0])
#print(series.info())
#print(series.index)
series_resample=series.resample(rule='T').mean()
#pyplot.plot(series_resample)
#pyplot.show()
one_m=timedelta(minutes=1)
for i in range(3):
    values = series_resample.values.astype('float32')
    n_steps = 5
    X, y = split_sequence(values, n_steps)
    #print(X.shape)
    X = X.reshape((X.shape[0], X.shape[1], 1)) 
    n_test = 12
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:] #Il '-' significa che prende gli ultimi valori del file csv. Io li modifico, modificando n_test.
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_steps,1)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2, validation_data=(X_test, y_test))
    # evaluate the model
    mse, mae = model.evaluate(X_test, y_test, verbose=0)
    print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
    # make a prediction
    row = y[-5:]
    #print(y[-5:])
    row_as = asarray(row).reshape((1, n_steps, 1))
    yhat = model.predict(row_as)
    print('Predicted: %.3f' % (yhat))
    pred=yhat[0,0]
    #today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    last_value_df=series_resample.index[-1]
    #last_value_convert=datetime.strptime(last_value_df, '%Y-%m-%d %H:%M:%S')
    new_date=last_value_df + one_m
    #print(new_date)
    series_resample.loc[new_date]=[pred]
    print(series_resample)

pyplot.plot(series_resample)
pyplot.show()






