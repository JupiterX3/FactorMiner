
def calculate(data, period=20, std_dev=2):
    """
    è®¡ç®å¸æå¸¦ææ ?
    
    åæ°:
        - period: å¨æ
        - std_dev: æ åå·®åæ°
    """
    # 1. è®¡ç®ä¸­è½¨(SMA)
    middle = data['close'].rolling(window=period).mean()
    
    # 2. è®¡ç®æ åå·?
    std = data['close'].rolling(window=period).std()
    
    # 3. è®¡ç®ä¸è½¨åä¸è½?
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # 4. è®¡ç®å¸¦å®½
    bandwidth = (upper - lower) / middle
    
    return bandwidth  # è¿åå¸¦å®½
