def calculate(data, supply_col='circulating_supply', price_col='close', **kwargs):
    import pandas as pd
    if supply_col in data.columns and price_col in data.columns:
        return (data[supply_col] * data[price_col]).fillna(float('nan'))
    return pd.Series(float('nan'), index=data.index)
