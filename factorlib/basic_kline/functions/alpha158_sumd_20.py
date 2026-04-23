import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return 2.0 * (ts_sum(delta(data['close'], 1).clip(lower=0), 20) / (ts_sum(delta(data['close'], 1).abs(), 20) + EPS)) - 1.0
