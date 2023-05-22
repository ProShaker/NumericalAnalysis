# https://youtu.be/QIUxPv5PJOY

# Description : This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM)
#               to predict the closing stock price of a corporation (Apple Inc) using the past 60 day stock price

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yfin

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
plt.style.use('fivethirtyeight')

company = 'UNH'
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2020, 1, 1)

# Get the stock quote
df = pdr.get_data_yahoo(company, start_date, end_date)

# Get the number of rows and columns in the data set
df.shape

# Visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()


