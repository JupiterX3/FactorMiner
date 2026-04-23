import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma20 = close.rolling(window=20).mean()
    ma50 = close.rolling(window=50).mean()
    return (ma20 - ma50) / ma50
