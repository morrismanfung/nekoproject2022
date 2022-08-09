# ネコ (Neko) Project

The ネコ Project attempts to predict stock price in the short future. We target SPY because we mainly invest on index fund only before we started the project. With more precise predictions, we plan to trade on short term (weekly) options to utilize the value of the predictions.

Using a Long Short Term Memory (LSTM) neural netowrk, we aim to predict the stock's close price on the next day (T+1). It is believed that price actions and volume changes can indicate some underlying factors affecting stock prices, such as the trading actions of investment banks. Thus, we are using price and volume in the previous 30 days (From T-29 to T) to predict the close price on the next day. For forecasting, we are shifting the time window to the future and recurrently using the new predictions as inputs for predictions on the further future (from T+2 to T+7).

*Note.* Another approach is to train an LSTM network to predict the target close price (i.e., price on T+7) directly with data in the previous days(From T-29 to T). We will test this approach in the future.

# Methodology

## Basic Principle

In this project, we need to import raw data of our targeted stock (i.e., SPY) and calculate some potentially useful trading indicators for the forecast. After converting the time series into appropriate arrays for LSTM, we train the models and determine the best sets of indicators used.

## Data importation

Price and volume of a stock ticker were improted with `yfinance`. In the current project, we are using data starting from Jan 1, 2018 till the current date (Aug 9, 2022).

The following code will automatically download the data of price and volume to the project.

```
import yfinance as yf
df_raw = yf.download( 'SPY', start = '2018-1-1')
```

## Trading indicators

To obtain trading indicators such as moving average (MA) or MACD (Moving average convergence divergence), we use the package `stockstats`. [stockstats](https://github.com/jealous/stockstats) introduced a wrapper StockDataFrame that is based on `pandas` data frame. It allows the fast calculation of the many technical indicators.

The code below returns a StockDataFrame with the close price, 10 days Exponential Moving Average (EMA), 20 days EMA, 10 days Simple Moving Average (SMA), 20 days SMA, volume, Relative Strength Index (RSI), MACD, Money Flow Index (MFI), and Average True Range (ATR). Indicators were picked arbitrarily as we do not have a strong background in finanace and investment.

```
from stockstats import wrap
wdf = wrap ( df_raw)
wdf[ ['close','close_10_ema', 'close_20_ema', 'close_10_sma', 'close_20_sma', 'volume', 'rsi', 'macd', 'mfi', 'atr']])
```

## Feature engineneering

We first standardized the data with `sklearn.preprocessing.MinMaxScaler`. Standardization is necessary because LSTM neural networks are sensitive to scales.

After standardization, it is essential to convert the time series vectors (one vector for each indicator) into matrixs and 3d arrays.

The resulting 3d array should have the dimension ( n_samples, n_steps, n_features), with *n_samples* as the number of days for prediction, *n_steps*, as number of days of data as input into the model (i.e., 30 in this project), and *n_features* as the number of features (i.e., depending on the number of technical indicator used.)

While some used a single function to create the 3d array from vectors, we adopt a simpler approach by altering vectors into matrixes and stacking matrixes into the final array in 2 separated steps.

The code to transform a vector into a matrix was modified from [https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

```
def to_matrix( tseries, steps):
    dataX, datay = [], []
    for i in range( len( tseries) - steps): # e.g. range( 48), i.e. 0 to 47 (inclusive)
        xs = tseries[ i:( i + steps)] # e.g. tseries[ 0:2] and tseries[ 47:49]. Getting n Xs (n = steps)
        dataX.append( xs) # adding the new xs into the list
        datay.append( tseries[ i + steps]) # adding the entry right after Xs into the list
    return np.array( dataX), np.array( datay)

indicator_x, indicator_y = to_matrix( df[ name_of_indicator], 30)
```

We apply the same function on all the technical indicators to create matrixes, 1 matrix for 1 indicator. Then we use `np.dstack` to stack all the matrixes in a 3d array.

## Model training

In the current project, we use a training size of 0.7. A simple sequential model with 1 LSTM hidden layer is used.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
model.add( LSTM( 20, input_shape = (30, n_features)))
model.add( Dense( 1))
model.compile( loss = 'mean_squared_error', optimizer = 'adam')

model.fit( X_train, y_train, epochs = 500, validation_data = (X_test, y_test))
```

# Results

## Indicators

We compared serveral models with different subsets of inputs:

| Model | Variables                                            |
|-------|------------------------------------------------------|
| 1     | Close price, Volume, EMAs, SMAs, RSI, MACD, MFI, ATR |
| 2     | Close price, Volume, RSI, MACD, MFI, ATR             |
| 3     | Close price, Volume, RSI, MACD, ATR                  |
| 4     | Close price, Volume, RSI, MACD                       |
| 5     | Close price, Volume, RSI                             |
| 6     | Close price, Volume                                  |

We compared model performance just by looking at the minimum validation loss (i.e., mean squared error, MSE) during the training processes. Model 1 with the moving average performs the worst. Model 2 and Model 3 perform similarly with an MSE of 0.00068. Model 4 does not perform well with an MSE of 0.00081. Model 5 and Model 6 perform the best with an MSE of 0.00045.

> *Note* This method of comparing model performance is not rigorous. We will use other method later in the future.

## Performance

Image of the testing set. As number of epoch is not finalized, the model should be overfitting, resulting in imperfect predictions.

![Performance](https://github.com/morrismanfung/nekoproject2022/blob/main/image/Figure_1.png)
