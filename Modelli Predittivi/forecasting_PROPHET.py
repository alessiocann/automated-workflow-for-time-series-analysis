# Python
import pandas as pd
from prophet import Prophet
from statsmodels.tools.eval_measures import rmse
from prophet.diagnostics import performance_metrics, cross_validation
from matplotlib import pyplot

df = pd.read_csv('nuovo_ping.csv')
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
m.plot(forecast)
pyplot.title("PROPHET")
pyplot.show()


