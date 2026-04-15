import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    return data['close'].rolling(window=10).quantile(0.75)
