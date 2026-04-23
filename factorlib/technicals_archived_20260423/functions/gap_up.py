def calculate(data, **kwargs):

    mode = kwargs.get('mode', 'open_vs_prev_close')  # 'open_vs_prev_close' | 'open_vs_prev_high' | 'strict'
    min_gap = float(kwargs.get('min_gap', 0.0))      # ç¸å¯¹éå¼ï¼ä¾å¦ 0.005 è¡¨ç¤º 0.5%
    return_type = kwargs.get('return_type', 'magnitude')  # 'magnitude' | 'bool'

    prev_close = data['close'].shift(1)
    prev_high = data['high'].shift(1)

    if mode == 'strict':
        # å½åæä½ä»·é«äºåä¸æ ¹æé«ä»·
        condition = data['low'] > prev_high
        gap_mag = (data['low'] - prev_high) / prev_high
    elif mode == 'open_vs_prev_high':
        # å¼çä»·é«äºåä¸æ ¹æé«ä»·
        condition = data['open'] > prev_high
        gap_mag = (data['open'] - prev_high) / prev_high
    else:
        # é»è®¤ï¼å¼çä»·é«äºåä¸æ ¹æ¶çä»·
        condition = data['open'] > prev_close
        gap_mag = (data['open'] - prev_close) / prev_close

    if return_type == 'bool':
        if min_gap > 0:
            out = (gap_mag > min_gap) & condition
        else:
            out = condition
        return out.astype('float64').reindex(data.index).fillna(0.0)
    else:
        # ä»ä¿çåä¸ç¼ºå£çæ­£å¹åº¦ï¼å¶ä»ç½?0
        mag = gap_mag.where(condition, 0.0)
        if min_gap > 0:
            mag = mag.where(mag > min_gap, 0.0)
        return mag.reindex(data.index).fillna(0.0)

