# automated-workflow-for-time-series-analysis
In this work is presented a system which automatically chooses the most suitable model, among
the available ones, to forecast the future values of a timeseries In particular, it was implemented a
workflow capable of getting monitoring data from the network (scraping using Prometheus), to
prepare data for the training of the forecasting models (LSTM, ARIMA and Prophet) and a selector
of the best model based on the RMSE (Root Mean Square Error).