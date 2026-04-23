import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return ts_std(data['volume'], 60) / (data['volume'] + EPS)
