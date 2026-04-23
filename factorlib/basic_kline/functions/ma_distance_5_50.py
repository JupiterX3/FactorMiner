import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma5 = close.rolling(window=5).mean()
    ma50 = close.rolling(window=50).mean()
    return (ma5 - ma50) / ma50
