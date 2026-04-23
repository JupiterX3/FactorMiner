import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    low = data['low']
    high = data['high']
    close = data['close']
    
    lowest_low = low.rolling(window=10).min()
    highest_high = high.rolling(window=10).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return stoch_k.rolling(window=3).mean()
