import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    if 'funding_rate' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    return data['funding_rate']
