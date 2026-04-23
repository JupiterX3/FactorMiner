import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    low_20 = close.rolling(window=20).min()
    return (close - low_20) / close
