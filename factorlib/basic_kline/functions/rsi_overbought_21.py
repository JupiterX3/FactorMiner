import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=21).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return (rsi > 70).astype(int)
