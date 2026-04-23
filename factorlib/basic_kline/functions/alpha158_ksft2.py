import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'] + EPS)
