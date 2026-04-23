def calculate(data, period=10, multiplier=3, **kwargs):
    """
    è®¡ç®Supertrendææ 
    
    é¿åæªæ¥å½æ°ï¼éæè®¡ç®é»è¾ï¼ç¡®ä¿ææè®¡ç®é½åºäºåå²æ°æ®
    ä½¿ç¨åéåæä½åæ»åå¤çï¼é¿åå¨æ¶é´tä½¿ç¨tæçä¿¡æ¯
    
    Args:
        data: åå«OHLCVæ°æ®çDataFrame
        period: ATRè®¡ç®å¨æï¼é»è®?0
        multiplier: ATRåæ°ï¼é»è®?
        **kwargs: å¶ä»åæ°
        
    Returns:
        Supertrendå¼Series
    """
    import pandas as pd
    import numpy as np
    
    high = data['high']
    low = data['low']
    close = data['close']
    
    # è®¡ç®çå®æ³¢å¹ (True Range)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # è®¡ç®ATRï¼å¹³åçå®æ³¢å¹ï¼
    atr = tr.rolling(window=period).mean()
    
    # è®¡ç®åºç¡ä¸ä¸è½?
    basic_upper = (high + low) / 2 + multiplier * atr
    basic_lower = (high + low) / 2 - multiplier * atr
    
    # åå§åæç»ä¸ä¸è½¨
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    
    # éæè®¡ç®é»è¾ï¼é¿åæªæ¥å½æ?
    # ä½¿ç¨shift(1)ç¡®ä¿å¨æ¶é´tåªä½¿ç¨t-1åä¹åçä¿¡æ¯
    for i in range(1, len(data)):
        # ä½¿ç¨æ»å1æçæ¶çä»·è¿è¡å¤æ?
        prev_close = close.iloc[i-1] if i > 0 else close.iloc[i]
        prev_upper = final_upper.iloc[i-1] if i > 0 else final_upper.iloc[i]
        prev_lower = final_lower.iloc[i-1] if i > 0 else final_lower.iloc[i]
        
        # æ´æ°ä¸è½¨
        if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_upper
        
        # æ´æ°ä¸è½¨
        if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_lower
    
    # è®¡ç®Supertrendå?
    supertrend = pd.Series(index=close.index, dtype=float)
    
    # åå§åç¬¬ä¸ä¸ªå?
    supertrend.iloc[0] = final_upper.iloc[0]
    
    # ä½¿ç¨æ»åé»è¾è®¡ç®åç»­å?
    for i in range(1, len(data)):
        prev_supertrend = supertrend.iloc[i-1]
        prev_upper = final_upper.iloc[i-1]
        prev_lower = final_lower.iloc[i-1]
        
        # ä½¿ç¨æ»å1æçæ¶çä»·è¿è¡å¤æ?
        current_close = close.iloc[i]
        
        if prev_supertrend == prev_upper:
            # å¦æåä¸æå¨ä¸è½¨ï¼æ£æ¥æ¯å¦è·ç ?
            if current_close <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            # å¦æåä¸æå¨ä¸è½¨ï¼æ£æ¥æ¯å¦çªç ?
            if current_close >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
    
    return supertrend

