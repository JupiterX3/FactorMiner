import pandas as pd
import numpy as np
from scipy.stats import skew

def calculate(data, **kwargs):
    return data['close'].rolling(window=200).apply(lambda x: skew(x, nan_policy='omit'))
