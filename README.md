# Time Series Forecasting in Python for VIX Index Prediction 

*In this project, I am going to build a basic time series forecasting model (ARIMA) to preidct the VIX Index using Python. Let's dive into it :)* 

### What is VIX?
The VIX or the Cboe Volatility Index is a real-time index that represents the market's expectations for the relative strength of near-term price changes of the S&P 500 index (SPX). It is an important index in the world of trading and investment because it provides a quantifiable measure of market risk and investors' sentiments.

### Time Series Analysis 
Import some packages
```
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
```
I got my data from yahoo finance. It is daily data from '2016-08-05' to '2021-08-04' (https://ca.finance.yahoo.com/quote/%5EVIX/history?p=%5EVIX)
```
df = pd.read_csv('VIXX.csv')
df = pd.DataFrame(df)
```



