import pandas as pd
import numpy as np

def calculate(data, period=20, std_dev=2, **kwargs):
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    bb_range = (upper - lower).replace(0, np.nan)
    position = (data['close'] - lower) / bb_range
    return position.fillna(0.5)
