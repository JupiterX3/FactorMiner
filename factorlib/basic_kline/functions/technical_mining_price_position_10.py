import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    low_10 = close.rolling(window=10).min()
    high_10 = close.rolling(window=10).max()
    return (close - low_10) / (high_10 - low_10)
