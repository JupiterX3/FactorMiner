import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    returns = close.pct_change()
    return (returns < 0).rolling(window=20).mean()
