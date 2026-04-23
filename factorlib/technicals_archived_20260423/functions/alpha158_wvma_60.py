import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return ts_std((data['close'] / (ref(data['close'], 1) + EPS) - 1.0).abs() * data['volume'], 60) / (ts_mean((data['close'] / (ref(data['close'], 1) + EPS) - 1.0).abs() * data['volume'], 60) + EPS)
