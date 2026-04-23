import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    window = 50
    close = data['close']
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return ((close - rolling_mean) / rolling_std).fillna(0.0)
