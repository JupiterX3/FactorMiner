import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma10 = close.rolling(window=10).mean()
    ma50 = close.rolling(window=50).mean()
    cross = (ma10 > ma50).astype(int)
    cross_signal = cross.diff()
    return cross_signal
