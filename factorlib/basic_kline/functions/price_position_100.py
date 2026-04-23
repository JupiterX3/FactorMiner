import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    low_100 = close.rolling(window=100).min()
    high_100 = close.rolling(window=100).max()
    return (close - low_100) / (high_100 - low_100)
