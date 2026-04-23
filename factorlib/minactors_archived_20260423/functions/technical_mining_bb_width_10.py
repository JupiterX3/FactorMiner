import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    ma = close.rolling(window=10).mean()
    std = close.rolling(window=10).std()
    return (2 * std) / ma
