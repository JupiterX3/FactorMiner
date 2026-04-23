import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return (data['close'] - ts_min(data['low'], 60)) / (ts_max(data['high'], 60) - ts_min(data['low'], 60) + EPS)
