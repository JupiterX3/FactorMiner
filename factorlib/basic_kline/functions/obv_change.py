import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
    return obv.pct_change()
