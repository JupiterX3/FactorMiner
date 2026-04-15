def calculate(data, period=10, multiplier=3, **kwargs):
    import pandas as pd
    from factorlib.technicals.functions.supertrend import calculate as supertrend_calc
    st = supertrend_calc(data, period=period, multiplier=multiplier)
    return (data['close'] >= st).astype('int').replace({0: -1, 1: 1})

