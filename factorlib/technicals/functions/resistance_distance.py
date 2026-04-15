import pandas as pd
import numpy as np

def calculate(data, lookback=50, **kwargs):
    resistance = data['high'].rolling(window=lookback).max()
    distance = (resistance - data['close']) / data['close'].replace(0, np.nan)
    return distance.fillna(0.0)
