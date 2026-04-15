def calculate(data, **kwargs):
    import pandas as pd
    import numpy as np
    bop = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    return bop.fillna(0.0)


