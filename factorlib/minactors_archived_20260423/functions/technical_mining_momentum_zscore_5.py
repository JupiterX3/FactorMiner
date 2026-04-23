import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    returns = close.pct_change()
    ma = returns.rolling(window=5).mean()
    std = returns.rolling(window=5).std()
    return (returns - ma) / std
