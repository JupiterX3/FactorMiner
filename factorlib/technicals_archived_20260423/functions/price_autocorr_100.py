import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    return data['close'].rolling(window=100).apply(lambda x: x.autocorr())
