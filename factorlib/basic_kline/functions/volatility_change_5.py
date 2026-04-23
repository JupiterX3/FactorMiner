import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    vol = returns.rolling(window=5).std()
    return vol / vol.shift(5)
