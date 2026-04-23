import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    open_price = data['open']
    close = data['close']
    low = data['low']
    
    body = close - open_price
    lower_shadow = np.minimum(open_price, close) - low
    
    long_lower_shadow = ((lower_shadow > body.abs() * 2) & (body > 0)).astype(int)
    return long_lower_shadow
