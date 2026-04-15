import pandas as pd
import numpy as np
from scipy.stats import skew

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    return returns.rolling(window=100).apply(lambda x: skew(x, nan_policy='omit'))
