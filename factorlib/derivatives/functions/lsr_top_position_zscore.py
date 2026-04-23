import pandas as pd
import numpy as np

def calculate(data, window=96, **kwargs):
    if 'lsr_top_position' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    x = data['lsr_top_position']
    return (x - x.rolling(window).mean()) / x.rolling(window).std()
