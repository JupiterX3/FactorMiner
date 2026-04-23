import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return ref(data['close'], 5) / (ref(data['close'], 60) + EPS)
