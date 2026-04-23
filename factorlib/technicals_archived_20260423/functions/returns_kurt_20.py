import pandas as pd
import numpy as np
from scipy.stats import kurtosis

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    return returns.rolling(window=20).apply(lambda x: kurtosis(x, nan_policy='omit'))
