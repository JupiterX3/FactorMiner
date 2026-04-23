import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    low = data['low']
    high = data['high']
    close = data['close']
    
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return (stoch_k < 20).astype(int)
