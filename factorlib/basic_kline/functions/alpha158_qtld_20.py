import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return ts_quantile(data['low'], 20, 0.2) / (data['close'] + EPS)
