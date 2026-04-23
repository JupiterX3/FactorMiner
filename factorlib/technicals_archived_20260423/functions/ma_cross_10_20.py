import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma10 = close.rolling(window=10).mean()
    ma20 = close.rolling(window=20).mean()
    cross = (ma10 > ma20).astype(int)
    cross_signal = cross.diff()
    return cross_signal
