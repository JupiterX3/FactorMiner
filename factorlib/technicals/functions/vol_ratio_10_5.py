import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    vol_5 = returns.rolling(window=5).std()
    vol_10 = returns.rolling(window=10).std()
    return vol_10 / vol_5
