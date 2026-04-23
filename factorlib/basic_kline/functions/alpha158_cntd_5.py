import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return (ts_mean((data['close'] > ref(data['close'], 1)).astype(float), 5) - ts_mean((data['close'] < ref(data['close'], 1)).astype(float), 5))
