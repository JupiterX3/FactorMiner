import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    if 'lsr_global_account' not in data.columns \
            or 'lsr_top_account' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    return data['lsr_global_account'] - data['lsr_top_account']
