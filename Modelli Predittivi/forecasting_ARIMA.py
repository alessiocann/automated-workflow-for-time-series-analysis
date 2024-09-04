# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse


# Read the AirPassengers dataset
series = pd.read_csv('nuovo_ping.csv', header=0, parse_dates=[0], index_col=[0])
series_resample=series.resample(rule='T').mean()
# Print the first five rows of the dataset
#series.head()


train = series_resample.iloc[:len(series_resample)-10]
test = series_resample.iloc[len(series_resample)-10:]
  
# Train the model on the full dataset
model = SARIMAX(train['Ping'], 
                        order = (0, 1, 1), 
                        seasonal_order =(2, 1, 1, 12))

result = model.fit()
  
# Forecast for the next 3 years
#forecast = result.predict(start = len(series), 
 #                         end = (len(series)-1) + 3 * 12, 
   #                       typ = 'levels').rename('Forecast')
 
forecast = result.predict(start = len(train), 
                          end = (len(train) + len(test) - 1), 
                          typ = 'levels').rename("Forecast")
                          

print(test["Ping"])
print(forecast)                         
# Calculate root mean squared error
print(rmse(test["Ping"], forecast))

forecast = result.predict(start = len(train), 
                          end = (len(train) + len(test) - 1) + 60, 
                          typ = 'levels').rename("Forecast")
  
  
# Plot the forecast values
series_resample['Ping'].plot(figsize = (12, 5), legend = True)
forecast.plot(legend = True)

plt.show()

  



