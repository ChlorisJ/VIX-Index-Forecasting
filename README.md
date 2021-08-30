# Time Series Forecasting in Python for VIX Index Prediction 

*In this project, I am going to build a basic time series forecasting model (ARIMA) to preidct the VIX Index using Python. Let's dive into it :)* 


## What is VIX?
The VIX or the Cboe Volatility Index is a real-time index that represents the market's expectations for the relative strength of near-term price changes of the S&P 500 index (SPX). It is an important index in the world of trading and investment because it provides a quantifiable measure of market risk and investors' sentiments.


## Visualizing the Data 

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


## Checking for Stationarity

When forecasting, most time series models assume that each point is independent of one another. The best indication of this is when the dataset of past instances is **stationary**. 

Any stationarity model will have:
 - Constant Mean
 - Constant Variance(The variations shouldn’t have visible trend/irregular)
 - No seasonality 

One approach I took was to split my data into chuncks and see each of their means & variances. 

![image](https://user-images.githubusercontent.com/77589878/130543807-5f56b0a1-7063-40b2-a28a-639920d10967.png)
> This is not too clear though...

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


## Model Building 

### Choosing Parameters 

ARIMA stands for AutoRegressive Integrated Moving Average, which is one of THE most commonly used model in time series forecasting. It is composed of 3 terms(Auto-Regression + Integrated + Moving-Average)

#### Auto-Regression

This is the AR(p) term of the model. It's showing we are using the previous **value** of the time series fore prediction. It means p lags of Y will be used as predictors. We can find the required number of AR terms by inspecting the Partial Autocorrelation (PACF) plot.

```
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(np.array(B))
```
![image](https://user-images.githubusercontent.com/77589878/130710421-50925978-d40c-4f9a-9f8b-bcf64b7b4252.png)

> We can see that lag 1 is significant as it is well above the significance line. Lag 2 is significant as well but is only slightly above the significance line. Hence I will be choosing p=1 for the AR term.

#### Moving Average

This is the MA(q) term of the model. It's showing we are using the previous **error** of the time series fore prediction. It means q lagged forecast errors will be used as predictors. We can find the required number of MA terms by inspecting the Autocorrelation (ACF) plot.

```
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(np.array(B))
```
![image](https://user-images.githubusercontent.com/77589878/130710877-1afbea5c-52f8-4731-ac4b-4e4b8c5d9be8.png)

> Hun... This is interesting because we can see that ACF plot is showing a 'decay' indicating that our time series may be non-stationary while the ADF test suggested otherwise. 

> It's time for some differencing experiments!

- First order differencing 
```
B1 = B.diff()
B1 = B1.dropna()
plot_acf(np.array(B1))
```
![image](https://user-images.githubusercontent.com/77589878/130711913-4591460e-3e48-4ca6-b64b-fa23261aec9d.png)

> Now this is what I was expecting! We can see that lag 1 is significant as it is above the significance line. Lag 2 is significant as well but is only slightly above the significance line. Hence I will be choosing q=1 for the MA term.

> Let's plot second order just to make sure.

- Second order differencing 
```
B2 = B.diff().diff()
B2 = B2.dropna()
plot_acf(np.array(B2))
```
![image](https://user-images.githubusercontent.com/77589878/130712005-d2e600b6-d837-4d3a-9322-ef4c0d08a273.png)

> This is an issue. See that the lag goes into the far negative zone fairly quick, which indicates the series might have been over-differenced.

#### Integrated

This is the differencing part of the model. Differencing the time series will make non-stationary data stationary. If differencing with previous value, its order 1 and so on.

> In conclusion, by looking at the ACF & PACF plots we can conclude our ARIMA would be ARIMA(1,1,1)

But there is a easier way to tell...
A simple function in python will allow for automatic selection for ARIMA based on each AICs

```
from pmdarima import auto_arima
stepwise_fit = auto_arima(B['Close'], trace=True,suppress_warnings=True)
```
```
Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=3160.266, Time=1.01 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=3184.591, Time=0.03 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=3175.880, Time=0.07 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=3173.876, Time=0.07 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=3182.619, Time=0.02 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=3171.263, Time=0.28 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=3171.827, Time=0.25 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=1.33 sec
 ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=1.23 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=3157.207, Time=0.33 sec
 ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=3171.211, Time=0.10 sec
 ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=3170.948, Time=0.09 sec
 ARIMA(1,1,1)(0,0,0)[0]             : AIC=3155.252, Time=0.15 sec
 ARIMA(0,1,1)(0,0,0)[0]             : AIC=3171.910, Time=0.04 sec
 ARIMA(1,1,0)(0,0,0)[0]             : AIC=3173.912, Time=0.03 sec
 ARIMA(2,1,1)(0,0,0)[0]             : AIC=3169.863, Time=0.12 sec
 ARIMA(1,1,2)(0,0,0)[0]             : AIC=3169.300, Time=0.15 sec
 ARIMA(0,1,2)(0,0,0)[0]             : AIC=3169.249, Time=0.05 sec
 ARIMA(2,1,0)(0,0,0)[0]             : AIC=3168.985, Time=0.04 sec
 ARIMA(2,1,2)(0,0,0)[0]             : AIC=3158.312, Time=0.39 sec

Best model:  ARIMA(1,1,1)(0,0,0)[0]          
Total fit time: 9.502 seconds
```

> It's the same result as my manual selection! We can now move onto model building.

### Test Train Split

Before training the model, let's split our data into train and test sets. Therefore we can make predictions on the test data and see how it performs.
In this analysis, my test set consists of the last 30 datapoints of my series 

```
train=B.iloc[:-30]
test=B.iloc[-30:]

print('Observations: %d' % (len(B.Close)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

```
```
Observations: 878
Training Observations: 848
Testing Observations: 30
```
To visualize the split.

```
train.plot()
test.plot()
```
![image](https://user-images.githubusercontent.com/77589878/131268970-2ab4437e-4256-4cb1-97a8-5b1651eb6355.png)


### Model Fitting
Model fitting turns out to be very easy in python with only some simple snippets of code

```
from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(train,order=(1,1,1))
model=model.fit()
model.summary()
```
```
Dep. Variable:	Close	No. Observations:	834
Model:	ARIMA(1, 1, 1)	Log Likelihood	-1504.070
Date:	Wed, 25 Aug 2021	AIC	3014.140
Time:	20:35:50	BIC	3028.315
Sample:	0	HQIC	3019.575
- 834		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
ar.L1	0.8449	0.028	29.793	0.000	0.789	0.901
ma.L1	-0.9473	0.020	-47.674	0.000	-0.986	-0.908
sigma2	2.1661	0.025	87.996	0.000	2.118	2.214
Ljung-Box (L1) (Q):	1.38	Jarque-Bera (JB):	85514.71
Prob(Q):	0.24	Prob(JB):	0.00
Heteroskedasticity (H):	2.14	Skew:	4.18
Prob(H) (two-sided):	0.00	Kurtosis:	51.93

```

> Note that here our Ljung-Box p value is large, this is good because the null hypothesis of the test is the model does not show lack of fit. A significant p-value in this test rejects the null hypothesis that the time series isn’t autocorrelated, which means that the residuals are independent which is what we want for the model to be correct.

> To confim, residuals are plotted. Note that we want the residual to look like white noise, so then we know there isn't much we can do on the modelling side.

```
residuals = pd.DataFrame(model.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.rcParams["figure.figsize"] = (20,10)
plt.show()
```
![image](https://user-images.githubusercontent.com/77589878/130881926-c3e2d3fa-1084-43a1-8e3c-01629f2c915e.png)

> We can see that our series does look like white noise, and is normally distributed around 0, we are good!


### Forecasting 

Now we can use our model for prediction in the test set

```
start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,typ='levels')
pred.index=index

plt.ylim([5, 40])
plt.plot(train,label='Train')
plt.plot(test, label='Test')
plt.plot(pred,label='Predict')
plt.legend(loc='upper left', fontsize=12)
plt.show()
```
![image](https://user-images.githubusercontent.com/77589878/131267938-bc2ee796-6430-4f70-b378-8da2b3c9c784.png)


Please note the index is generated with code below. I had to manually pluck out some holidays for the indexes to match. 

```
index=pd.bdate_range(start=pd.to_datetime('2019-12-18'),end=pd.to_datetime('01/31/2020'))
index = index[index_future_dates != '2019-12-25' ]
index = index[index != '2020-01-01']
index = index[index != '2020-01-20']
index
```

Let's zoom in on our prediction!

![image](https://user-images.githubusercontent.com/77589878/131267948-06e8fd21-9470-43d8-8928-d88967231953.png)

> Overall this is not bad, my prediction is at the middle of the test set. However we can see that it is having some trouble fitting the spike at the end. Let's check model performance


### Model Performance 
For my model performance metric I am using MAPE. It is scale-independent in percentage so they can be used to compare forecasts between different series

```
Pred = pred.values
Test = test.values.T
mape = np.mean(np.abs(Pred - Test)/np.abs(Test)) 
mape
```
```
.07536801502900345
```

> Not bad I'd say! A .0753 MAPE (7.53%) means that my model is about 92.47% accurate in predicting the next 30 observations.

