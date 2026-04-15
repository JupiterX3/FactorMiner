import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    open_price = data['open']
    close = data['close']
    high = data['high']
    low = data['low']
    
    body = abs(close - open_price)
    range_val = high - low
    
    doji = (body / range_val < 0.1).astype(int)
    return doji
