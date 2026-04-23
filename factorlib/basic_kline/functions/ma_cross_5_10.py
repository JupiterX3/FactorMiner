import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma5 = close.rolling(window=5).mean()
    ma10 = close.rolling(window=10).mean()
    cross = (ma5 > ma10).astype(int)
    cross_diff = ma5 - ma10
    cross_signal = cross.diff()
    return cross_signal
