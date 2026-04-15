import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma = close.rolling(window=100).mean()
    std = close.rolling(window=100).std()
    return (close - ma) / std
