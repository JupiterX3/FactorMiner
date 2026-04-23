import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    vc = ts_sum(data['volume'] * data['close'], 5)
    v  = ts_sum(data['volume'], 5)
    return vc / (v + EPS)
