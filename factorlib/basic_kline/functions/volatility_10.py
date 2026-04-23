import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    return returns.rolling(window=10).std()
