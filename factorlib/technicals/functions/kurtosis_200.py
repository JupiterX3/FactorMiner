import pandas as pd
import numpy as np
from scipy.stats import kurtosis

def calculate(data, **kwargs):
    return data['close'].rolling(window=200).apply(lambda x: kurtosis(x, nan_policy='omit'))
