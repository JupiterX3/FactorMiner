def calculate(data, period=20, **kwargs):
    import pandas as pd
    # è¿periodåçå±é¨æé«ä½ä¸ºé»åä¼°è®?
    return data['high'].rolling(window=period).max().fillna(0.0)

