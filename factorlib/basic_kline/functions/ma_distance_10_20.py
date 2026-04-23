import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma10 = close.rolling(window=10).mean()
    ma20 = close.rolling(window=20).mean()
    return (ma10 - ma20) / ma20
