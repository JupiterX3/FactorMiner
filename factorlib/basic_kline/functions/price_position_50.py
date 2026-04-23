import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    low_50 = close.rolling(window=50).min()
    high_50 = close.rolling(window=50).max()
    return (close - low_50) / (high_50 - low_50)
