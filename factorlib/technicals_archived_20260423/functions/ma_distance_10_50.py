import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma10 = close.rolling(window=10).mean()
    ma50 = close.rolling(window=50).mean()
    return (ma10 - ma50) / ma50
