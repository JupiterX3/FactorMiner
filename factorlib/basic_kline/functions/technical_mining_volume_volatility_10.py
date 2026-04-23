import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    volume = data['volume']
    return volume.pct_change().rolling(window=10).std()
