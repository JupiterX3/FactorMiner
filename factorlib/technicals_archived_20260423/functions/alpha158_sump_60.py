import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return ts_sum(delta(data['close'], 1).clip(lower=0), 60) / (ts_sum(delta(data['close'], 1).abs(), 60) + EPS)
