import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return ts_corr(data['close'] / (ref(data['close'], 1) + EPS), np.log(data['volume'] / (ref(data['volume'], 1) + EPS) + 1.0), 10)
