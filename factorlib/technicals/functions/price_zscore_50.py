import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma = close.rolling(window=50).mean()
    std = close.rolling(window=50).std()
    return (close - ma) / std
