import datetime as dt
from datetime import datetime
import time
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy
from numpy import sqrt
from numpy import asarray
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
from prophet import Prophet
from prophet.diagnostics import performance_metrics, cross_validation
from prometheus_api_client import PrometheusConnect,  MetricSnapshotDataFrame, MetricRangeDataFrame




def scrape():
    prom = PrometheusConnect(url ="http://172.17.0.1:9090", disable_ssl=True)

    metric_data = prom.get_metric_range_data(
        metric_name='ping_avg',
        start_time=(dt.datetime.now() - dt.timedelta(hours=1)),
        end_time=dt.datetime.now()
    )
    metric_df = MetricRangeDataFrame(metric_data)
    del metric_df['__name__']
    del metric_df['instance']
    del metric_df['job']
    metric_df.reset_index('timestamp',inplace=True)
    metric_df.columns = ['timestamp', 'Ping']
    metric_df['Time']=pd.to_datetime(metric_df['timestamp'],unit='s')
    del metric_df['timestamp']
    metric_df.set_index('Time',inplace=True)
    print(metric_df)
    metric_df.to_csv("/graph/ping.csv")



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




def lstm(ti):
    series = pd.read_csv('/graph/ping.csv', header=0, parse_dates=[0], index_col=[0]) #Vuole il percorso assoluto del csv
    #print(series.info())
    #print(series.index)
    series_resample=series.resample(rule='T').mean()
    #pyplot.plot(series_resample)
    #pyplot.show()
    one_m=timedelta(minutes=1)
    for i in range(60):
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
        rmse=sqrt(mse)
        print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, rmse, mae))
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
    ti.xcom_push(key='model_accuracy', value=rmse)
    pyplot.plot(series_resample)
    pyplot.title("LSTM")
    pyplot.savefig("/graph/LSTM.png")
    


def arima(ti):
    # Read the AirPassengers dataset
    series = pd.read_csv('/graph/ping.csv', header=0, parse_dates=[0], index_col=[0])
    series_resample=series.resample(rule='T').mean()
    # Print the first five rows of the dataset
    #series.head()
    train = series_resample.iloc[:len(series_resample)-12]
    test = series_resample.iloc[len(series_resample)-12:]
    # Train the model on the full dataset
    model = SARIMAX(train['Ping'], 
                            order = (0, 1, 1), 
                            seasonal_order =(2, 1, 1, 12))
    result = model.fit()
    forecast = result.predict(start = len(train), 
                              end = (len(train) + len(test) - 1), 
                              typ = 'levels').rename("Forecast")
    print(test["Ping"])
    print(forecast)                         
    # Calculate root mean squared error
    value_rmse=rmse(test["Ping"], forecast)
    print(value_rmse)
    forecast = result.predict(start = len(train), 
                              end = (len(train) + len(test) - 1) + 60, 
                              typ = 'levels').rename("Forecast")
    ti.xcom_push(key='model_accuracy', value=value_rmse)
    # Plot the forecast values
    series_resample['Ping'].plot(figsize = (12, 5), legend = True)
    forecast.plot(legend = True)
    pyplot.title("ARIMA")
    pyplot.savefig("/graph/ARIMA.png")
 


def prophet(ti):
    df = pd.read_csv('/graph/ping.csv')
    df.columns = ['ds', 'y']
    #df['ds']= to_datetime(df['ds'])
    print(df.head())
    m = Prophet()
    m.fit(df)
    df_cv = cross_validation(m, horizon = '5 min')
    df_p = performance_metrics(df_cv)
    print(df_p.head())
    future = m.make_future_dataframe(periods=60, freq='min')
    print(future.tail())
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    df.columns = ['Time', 'Ping']
    rmse=df_p['rmse'].values[-1]
    ti.xcom_push(key='model_accuracy', value=rmse)
    m.plot(forecast)
    pyplot.title("PROPHET")
    pyplot.savefig("/graph/PROPHET.png")
 
   
    

def confrontoRSME(ti):
    rmse_LSTM = ti.xcom_pull(key='model_accuracy', task_ids=['forecasting_LSTM'])
    rmse_ARIMA = ti.xcom_pull(key='model_accuracy', task_ids=['forecasting_ARIMA'])
    rmse_PROPHET = ti.xcom_pull(key='model_accuracy', task_ids=['forecasting_PROPHET'])
    print("LSTM: ", rmse_LSTM)
    print("ARIMA: ", rmse_ARIMA)
    print("PROPHET: ", rmse_PROPHET)
    if rmse_LSTM < rmse_ARIMA and rmse_LSTM < rmse_PROPHET:
        print("Il modello più attendibile e' LSTM: ", rmse_LSTM)
    elif rmse_ARIMA < rmse_LSTM and rmse_ARIMA < rmse_PROPHET:
         print("Il modello più attendibile e' ARIMA: ", rmse_ARIMA)
    else:
        print("Il modello più attendibile e' PROPHET: ", rmse_PROPHET)
    

    

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 5, 8),
    'concurrency': 1,
    'retries': 0
}



with DAG('forecasting_PING', catchup=False, default_args=default_args, schedule_interval='*/30 * * * *',) as dag:
    opr_start = BashOperator(task_id='start',
                             bash_command='echo "START"')
                             
    opr_prometheus = PythonOperator(task_id='scrape_data',
                               python_callable=scrape)

    opr_lstm = PythonOperator(task_id='forecasting_LSTM',
                               python_callable=lstm)
    
    opr_arima = PythonOperator(task_id='forecasting_ARIMA',
                                 python_callable=arima)
                                 
    
    opr_prophet = PythonOperator(task_id='forecasting_PROPHET',
                                 python_callable=prophet)
                                 
    opr_confrontoRSME = PythonOperator(task_id='confronto_RSME',
                                 python_callable=confrontoRSME)
                                 
    opr_end = BashOperator(task_id='end',
                             bash_command='echo "END"')

opr_start >> opr_prometheus >> [opr_lstm, opr_arima, opr_prophet] >> opr_confrontoRSME >> opr_end



