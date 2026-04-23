import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    high = data['high']
    low = data['low']
    close = data['close']
    period = 21

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    atr_ratio = atr / close
    return atr_ratio
