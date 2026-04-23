import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return (data[['open','close']].min(axis=1) - data['low']) / (data['open'] + EPS)
