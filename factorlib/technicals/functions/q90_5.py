import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    return data['close'].rolling(window=5).quantile(0.90)
