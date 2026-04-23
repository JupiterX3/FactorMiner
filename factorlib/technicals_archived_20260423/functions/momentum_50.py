import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    return close.pct_change(periods=50)
