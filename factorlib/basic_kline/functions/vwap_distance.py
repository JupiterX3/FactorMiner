import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    vwap = (close * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    return (close - vwap) / vwap
