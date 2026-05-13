def calculate(data, period=10, multiplier=3, **kwargs):
    import pandas as pd
    import numpy as np

    high = data['high']
    low = data['low']
    close = data['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()

    basic_upper = (high + low) / 2 + multiplier * atr
    basic_lower = (high + low) / 2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    first_valid_idx = atr.first_valid_index()
    if first_valid_idx is None:
        return pd.Series(np.nan, index=close.index)
    first_valid = close.index.get_loc(first_valid_idx)

    for i in range(first_valid + 1, len(data)):
        prev_close = close.iloc[i - 1]
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]

        if np.isnan(prev_upper):
            final_upper.iloc[i] = basic_upper.iloc[i]
        elif basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_upper

        if np.isnan(prev_lower):
            final_lower.iloc[i] = basic_lower.iloc[i]
        elif basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_lower

    supertrend = pd.Series(np.nan, index=close.index, dtype=float)

    supertrend.iloc[first_valid] = final_upper.iloc[first_valid]

    for i in range(first_valid + 1, len(data)):
        prev_supertrend = supertrend.iloc[i - 1]
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]
        current_close = close.iloc[i]

        if np.isnan(prev_supertrend) or np.isnan(prev_upper) or np.isnan(prev_lower):
            continue

        if prev_supertrend == prev_upper:
            if current_close <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if current_close >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]

    return supertrend
