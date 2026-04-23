import pandas as pd
import numpy as np

def calculate(data, period=12, **kwargs):
    if 'open_interest' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    return data['open_interest'].pct_change(period)
