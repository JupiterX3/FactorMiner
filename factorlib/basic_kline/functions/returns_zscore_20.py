import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    ma = returns.rolling(window=20).mean()
    std = returns.rolling(window=20).std()
    return (returns - ma) / std
