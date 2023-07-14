import pandas as pd
import numpy as np

"""This is an file for experimenting with producing technical indicators from the market data."""

data = pd.read_csv('./data/binance.csv', index_col='date', parse_dates=True)

# print(data.head())
# print(TA.BBANDS(data))


"""
ROC (Rate of Change)
Simple percentage difference between current price and some previous price.
"""
# Smooth data with triple exponential moving average (TEMA)
# TEMA = 3(ema1) - 3(ema2) + ema3
smooth_period = 14
data['ema1'] = data['close'].ewm(span=smooth_period, adjust=False).mean()
data['ema2'] = data['ema1'].ewm(span=smooth_period, adjust=False).mean()
data['ema3'] = data['ema2'].ewm(span=smooth_period, adjust=False).mean()
data['tema'] = (3 * data['ema1']) - (3 * data['ema2']) + data['ema3']

# ROC = current_price - previous_price / previous_price
data['ROC2'] = data['tema'].pct_change(periods=-2) * 100
data['ROC4'] = data['tema'].pct_change(periods=-4) * 100
data['ROC8'] = data['tema'].pct_change(periods=-8) * 100
data['ROC16'] = data['tema'].pct_change(periods=-16) * 100
data['ROC32'] = data['tema'].pct_change(periods=-32) * 100

"""
ADX (Average Directional Index)
The smoothed average of DX (directional index).
DX = 100 * abs(+DI - -DI / +DI + -DI)
DI = 100 * smoothed(DM) / smoothed(TR), DM can be +DM or -DM
TR = 
To calculate DM, first calculate UpMove and DownMove
UpMove = current_high - prev_high  
DownMove = prev_low - curr_low
If UpMove > DownMove and UpMove > 0, +DM = UpMove, else +DM = 0
If DownMove > UpMove and DownMove > 0, -DM = DownMove, else -DM = 0
"""
# Calculate TR (true range)
data['high-low'] = data['high'] - data['low']
data['high-prev_close'] = abs(data['high'] - data['close'].shift(-1))
data['low-prev_close'] = abs(data['low'] - data['close'].shift(-1))
data['TR'] = data[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)

# Calculate DM (directional movement)
data['up_move'] = data['high'] - data['high'].shift(-1)
data['down_move'] = data['low'].shift(-1) - data['low']
data['pos_DM'] = data['up_move'].where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), 0)

data['neg_DM'] = data['down_move'].where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), 0)

print(data['pos_DM'], data['neg_DM'])

data.to_csv('data/binance.csv')
