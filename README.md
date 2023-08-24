# Stock Price Prediction using LSTM deep learning network

Data used from stock: <br> **Tata Consumer Products Ltd** <br> **NSE: TATACONSUM**


## Features

+ ***Date***: Date of the trading day
+ ***Open***: The opening price for stocks at start of the day
+ ***High***: Highest stock price for the day
+ ***Low***: Lowest stock price for the day
+ ***Last***: Price at which the last transaction for a stock went through
+ ***Close***: The price stock ended at for the day
+ ***Total Trade Quantity***: Stocks traded for the day
+ ***Turnover***: Turnover for the day

## Metrics

Loss: Mean Squared Error <br>
Metrics: Root Mean Squared Error, Mean Absolute Percent Error

## Results

Root Mean Squared Error: `7.873`<br>
Mean Absolute Percent Error: `3.02%`

## Note

_Model was built using Keras deep learning LSTM network, using 8 layers and 292621 trainable parameters_<br>
_Graphical representation of the model and predictions are available in `results/` directory_
