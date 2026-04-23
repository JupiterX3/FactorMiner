import pandas as pd
import numpy as np

def calculate(data, period=20, **kwargs):
    ret = data['close'].pct_change()
    return ret.rolling(window=period).kurt()
