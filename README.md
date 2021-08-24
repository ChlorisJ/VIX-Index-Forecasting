# Time Series Forecasting in Python for VIX Index Prediction 

*In this project, I am going to build a basic time series forecasting model (ARIMA) to preidct the VIX Index using Python. Let's dive into it :)* 


### What is VIX?
The VIX or the Cboe Volatility Index is a real-time index that represents the market's expectations for the relative strength of near-term price changes of the S&P 500 index (SPX). It is an important index in the world of trading and investment because it provides a quantifiable measure of market risk and investors' sentiments.


### Visualizing the data 

```
df = pd.read_csv('VIXX.csv')
df = pd.DataFrame(df)
```
I got my data from yahoo finance. It is daily data from '2016-08-05' to '2021-08-04' (https://ca.finance.yahoo.com/quote/%5EVIX/history?p=%5EVIX)
 
```
sns.lineplot(data=df['Close'])
```

I picked the index **closing** price for this analysis

![image](https://user-images.githubusercontent.com/77589878/130539367-d9513cea-4c8d-4bea-a42e-73730b77cec8.png)


> In absolute terms, VIX values greater than 30 are generally linked to large volatility resulting from increased uncertainty, risk, and investors' fear. VIX values below 20 generally correspond to stable, stress-free periods in the markets.

> We can see that for March 2020, the VIX index has spiked due to the novel pandemic which corresponds to hightened investors' fear and increased uncertainty. It has seen slow decrease going forward till Aug 2021, yet is still relatively higher than pre-pandemic level. Due to the observations since and after the pandemic being clear outliers, I belive it is better to build model leaving out those observations.

Removing pandemic and post-pandemic observations 
```
dat = df['Close'][df.index <= '2020-02-01']
B = pd.DataFrame(dat)
```


### Checking For Stationarity

When forecasting, most time series models assume that each point is independent of one another. The best indication of this is when the dataset of past instances is **stationary**. 

Any stationarity model will have:
 - Constant Mean
 - Constant Variance(The variations shouldn’t have visible trend/irregular)
 - No seasonality 

One approach I took was to split my data into chuncks and see each of their means & variances. 

![image](https://user-images.githubusercontent.com/77589878/130543807-5f56b0a1-7063-40b2-a28a-639920d10967.png)
> This is visually exhausting me LOL

A much easier way to check for stationarity called the ADF Test (Augmented Dickey Fuller Test).
```
result  = ts.adfuller(B, 1)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```
```
ADF Statistic: -5.135575
p-value: 0.000012
Critical Values:
	1%: -3.438
	5%: -2.865
	10%: -2.569
```
> Here the p-value is If 0.000012 < 0.05 suggesting the data is stationary. Perfect! We can go straight to model building


### Model Building 

#### ARIMA 

ARIMA stands for AutoRegressive Integrated Moving Average, which is one of THE most commonly used model in time series forecasting. It is composed of 3 terms(Auto-Regression + Integrated + Moving-Average)

#### Auto-Regression
This is the AR(p) term of the model. Basically showing we are using the previous value of the time series fore prediction. It means p lags of Y will be used as predictors. 
