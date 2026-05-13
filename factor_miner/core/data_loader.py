"""
数据获取模块
支持多种数据源的数据获取和预处理
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    数据加载器
    支持多种数据源的数据获取
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.cache = {}
        
    def get_data(self, 
                 symbol: str,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 data_source: str = 'yahoo',
                 interval: str = '1d',
                 **kwargs) -> pd.DataFrame:
        """
        获取市场数据
        
        Args:
            symbol: 股票代码或交易对
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据源 ('yahoo', 'binance', 'alpha_vantage')
            interval: 数据间隔
            **kwargs: 其他参数
            
        Returns:
            市场数据DataFrame
        """
        if data_source == 'yahoo':
            return self._get_yahoo_data(symbol, start_date, end_date, interval, **kwargs)
        elif data_source == 'binance':
            return self._get_binance_data(symbol, start_date, end_date, interval, **kwargs)
        elif data_source == 'alpha_vantage':
            return self._get_alpha_vantage_data(symbol, start_date, end_date, **kwargs)
        else:
            return self._get_local_exchange_data(symbol, start_date, end_date, interval, data_source, **kwargs)
    
    def _get_yahoo_data(self, 
                        symbol: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        interval: str = '1d',
                        **kwargs) -> pd.DataFrame:
        """
        从Yahoo Finance获取数据
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                **kwargs
            )
            
            # 标准化列名
            data.columns = [col.lower() for col in data.columns]
            
            # 添加技术指标
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            print(f"获取Yahoo数据失败: {e}")
            return pd.DataFrame()
    
    def _get_binance_data(self,
                          symbol: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          interval: str = '1d',
                          **kwargs) -> pd.DataFrame:
        """
        从本地币安数据文件获取数据
        """
        try:
            # 构建本地文件路径
            data_dir = Path(__file__).parent.parent.parent  / "data" / "binance" / "futures"
            # data_dir = Path("data/binance/futures")
            if not data_dir.exists():
                print(f"数据目录不存在: {data_dir}")
                return pd.DataFrame()
            
            # 查找匹配的数据文件
            # 文件名格式: SYMBOL_USDT_USDT-TIMEFRAME-futures.feather
            # 注意：symbol参数可能已经包含了_USDT，所以需要处理
            if symbol.endswith('_USDT'):
                base_symbol = symbol
            else:
                base_symbol = f"{symbol}_USDT"
            
            pattern = f"{base_symbol}_USDT-{interval}-futures.feather"
            matching_files = list(data_dir.glob(pattern))
            
            print(f"查找模式: {pattern}")
            print(f"找到文件: {[f.name for f in matching_files]}")
            
            if not matching_files:
                print(f"未找到匹配的数据文件: {pattern}")
                return pd.DataFrame()
            
            # 读取第一个匹配的文件
            file_path = matching_files[0]
            print(f"读取数据文件: {file_path}")
            
            data = pd.read_feather(file_path)
            
            # 确保有日期索引
            if 'date' in data.columns:
                data.set_index('date', inplace=True)
            elif 'time' in data.columns:
                data.set_index('time', inplace=True)
            elif 'timestamp' in data.columns:
                data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
                data.set_index('date', inplace=True)
                data.drop('timestamp', axis=1, inplace=True)
            
            # 过滤日期范围
            if start_date:
                start_dt = pd.to_datetime(start_date)
                data = data[data.index >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                data = data[data.index <= end_dt]
            
            print(f"成功加载数据: {data.shape[0]} 条记录")
            return data

        except Exception as e:
            print(f"获取币安数据失败: {e}")
            return pd.DataFrame()
    
    # ------------------------------------------------------------------
    # 额外数据列 join（metrics / funding / mark / index）
    # ------------------------------------------------------------------
    @staticmethod
    def _left_join_on_date(main_df: pd.DataFrame, extra_df: pd.DataFrame,
                           how: str = 'left') -> pd.DataFrame:
        if extra_df is None or extra_df.empty:
            return main_df
        if main_df is None or main_df.empty:
            return main_df

        def _normalize_dt_index(idx) -> pd.DatetimeIndex:
            dt_idx = pd.to_datetime(idx, errors='coerce')
            # 统一到 UTC-naive，避免 tz-aware 与 tz-naive 之间 reindex 报 dtype 不可比较
            try:
                if getattr(dt_idx, 'tz', None) is not None:
                    dt_idx = dt_idx.tz_convert('UTC').tz_localize(None)
            except Exception:
                try:
                    dt_idx = dt_idx.tz_localize(None)
                except Exception:
                    pass
            return dt_idx

        extra = extra_df.copy()
        if 'date' in extra.columns:
            extra['date'] = pd.to_datetime(extra['date'], errors='coerce')
            extra = extra.set_index('date')

        # 临时将主表索引和 extra 索引都规范化到同一时区口径进行对齐，
        # 完成后恢复主表原始索引，避免影响调用方。
        original_main_index = main_df.index
        main = main_df.copy()
        main.index = _normalize_dt_index(main.index)
        extra.index = _normalize_dt_index(extra.index)

        extra = extra[extra.index.notna()]
        extra = extra[~extra.index.duplicated(keep='last')].sort_index()

        aligned = extra.reindex(main.index, method='ffill')
        for col in aligned.columns:
            main[col] = aligned[col]

        main.index = original_main_index
        return main

    def join_extras(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str = '1h',
        include: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        向已有 DataFrame 按 date 索引左连接额外数据列（metrics / funding / mark / index / basis / macro / sentiment）。

        适用于已经通过 load_local_market_data 等方式加载了基础 OHLCV 的场景，
        只需追加衍生品 / 资金费率等额外列即可。

        Args:
            df: 基础 OHLCV DataFrame（需有 DatetimeIndex）。
            symbol: 交易对，如 ``BTC_USDT``。
            interval: K 线周期，用于定位 mark / index / macro / sentiment 文件。
            include: 要 join 的额外数据类别，支持 ``'metrics'``、``'funding'``、
                     ``'mark'``、``'index'``、``'basis'``（快捷：自动含 mark+index 并派生）、
                     ``'macro'``（全局宏观因子 DXY/SPX/IXIC/Gold/10Y）、
                     ``'sentiment'``（全局情绪因子 VIX/稳定币购买力）。
                     默认 None 等价于 ``['metrics', 'funding', 'basis']``。

        Returns:
            追加了额外列的 df（原地修改并返回）。找不到的额外文件不会报错，只是对应列缺失。
        """
        if df is None or df.empty:
            return df

        include = include if include is not None else ['metrics', 'funding', 'basis']

        safe_symbol = symbol if symbol.endswith('_USDT_USDT') else (
            f"{symbol}_USDT_USDT" if '_USDT' not in symbol else f"{symbol}_USDT"
        )

        project_root = Path(__file__).parent.parent.parent
        base_dir = project_root / "data" / "binance"

        # 1. metrics（归档固定 5m 粒度，由 _left_join_on_date ffill 到目标周期）
        if 'metrics' in include:
            metrics_path = base_dir / "futures_metrics" / (
                f"{safe_symbol}-5m-metrics.feather"
            )
            if metrics_path.exists():
                try:
                    df = self._left_join_on_date(df, pd.read_feather(metrics_path))
                except Exception as e:
                    print(f"join metrics 失败: {e}")

        # 2. funding（事件级，需要 ffill 到 K 线）
        if 'funding' in include:
            funding_path = base_dir / "futures_funding" / f"{safe_symbol}-funding.feather"
            if funding_path.exists():
                try:
                    df = self._left_join_on_date(df, pd.read_feather(funding_path))
                except Exception as e:
                    print(f"join funding 失败: {e}")

        # 3. mark / index / basis
        want_basis = 'basis' in include
        need_mark = 'mark' in include or want_basis
        need_index = 'index' in include or want_basis
        if need_mark:
            mark_path = base_dir / "futures_markprice" / (
                f"{safe_symbol}-{interval}-mark.feather"
            )
            if mark_path.exists():
                try:
                    df = self._left_join_on_date(df, pd.read_feather(mark_path))
                except Exception as e:
                    print(f"join mark 失败: {e}")
        if need_index:
            index_path = base_dir / "futures_indexprice" / (
                f"{safe_symbol}-{interval}-index.feather"
            )
            if index_path.exists():
                try:
                    df = self._left_join_on_date(df, pd.read_feather(index_path))
                except Exception as e:
                    print(f"join index 失败: {e}")
        if want_basis and 'index_close' in df.columns:
            if 'mark_close' in df.columns:
                df['basis'] = (df['mark_close'] - df['index_close']) / df['index_close']
            elif 'close' in df.columns:
                df['basis'] = (df['close'] - df['index_close']) / df['index_close']

        # 5. macro（全局数据，不含 symbol）
        if 'macro' in include:
            macro_path = base_dir / "futures_macro" / f"{interval}-macro.feather"
            if macro_path.exists():
                try:
                    df = self._left_join_on_date(df, pd.read_feather(macro_path))
                except Exception as e:
                    print(f"join macro 失败: {e}")

        # 6. sentiment（全局数据，不含 symbol）
        if 'sentiment' in include:
            sentiment_path = base_dir / "futures_sentiment" / f"{interval}-sentiment.feather"
            if sentiment_path.exists():
                try:
                    df = self._left_join_on_date(df, pd.read_feather(sentiment_path))
                except Exception as e:
                    print(f"join sentiment 失败: {e}")

        return df

    def load_with_extras(
        self,
        symbol: str,
        interval: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include: Optional[List[str]] = None,
        trade_type: str = 'futures',
    ) -> pd.DataFrame:
        """
        加载主 OHLCV 并按需 join 额外数据列。

        Args:
            include: 子集，支持 'metrics'（OI/LSR/taker_ratio）、'funding'、
                     'mark'（mark 价 → basis 需同时 include 'index'）、'index'。
                     默认 None 等价于 ['metrics','funding','basis']。
                     'basis' 是快捷选项：自动包含 'mark'+'index' 并派生
                     `basis = (close - index_close) / index_close`。

        Returns:
            带有额外列的 OHLCV DataFrame（date 为索引）。找不到的额外文件
            不会报错，只是对应列缺失。
        """
        include = include if include is not None else ['metrics', 'funding', 'basis']

        base = self._get_binance_data(symbol, start_date, end_date, interval)
        if base is None or base.empty:
            return pd.DataFrame()

        return self.join_extras(base, symbol, interval=interval, include=include)

    def _get_local_exchange_data(self,
                                  symbol: str,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  interval: str = '1d',
                                  exchange: str = 'binance',
                                  **kwargs) -> pd.DataFrame:
        try:
            base_dir = Path(__file__).parent.parent.parent / "data"

            search_dirs = [
                base_dir / exchange / "futures",
                base_dir / exchange / "spot",
                base_dir / exchange,
                base_dir / "binance" / "futures",
                base_dir / "binance" / "spot",
                base_dir / "binance",
            ]

            if symbol.endswith('_USDT'):
                base_symbol = symbol
            else:
                base_symbol = f"{symbol}_USDT"

            for data_dir in search_dirs:
                if not data_dir.exists():
                    continue

                patterns = [
                    f"{base_symbol}_USDT-{interval}-futures.feather",
                    f"{base_symbol}_USDT-{interval}-spot.feather",
                    f"{base_symbol}_USDT-{interval}.feather",
                    f"{base_symbol}-{interval}-futures.feather",
                    f"{base_symbol}-{interval}-spot.feather",
                    f"{base_symbol}-{interval}.feather",
                ]

                for pattern in patterns:
                    matching_files = list(data_dir.glob(pattern))
                    if matching_files:
                        file_path = matching_files[0]
                        print(f"找到本地数据文件: {file_path}")
                        data = pd.read_feather(file_path)

                        if 'date' in data.columns:
                            data.set_index('date', inplace=True)
                        elif 'time' in data.columns:
                            data.set_index('time', inplace=True)
                        elif 'timestamp' in data.columns:
                            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
                            data.set_index('date', inplace=True)
                            data.drop('timestamp', axis=1, inplace=True)

                        if start_date:
                            start_dt = pd.to_datetime(start_date)
                            data = data[data.index >= start_dt]
                        if end_date:
                            end_dt = pd.to_datetime(end_date)
                            data = data[data.index <= end_dt]

                        print(f"成功加载本地数据: {data.shape[0]} 条记录")
                        return data

            print(f"未找到 {exchange} 交易所 {symbol} 的本地数据文件")
            return pd.DataFrame()

        except Exception as e:
            print(f"获取本地交易所数据失败: {e}")
            return pd.DataFrame()
    
    def _get_alpha_vantage_data(self,
                                symbol: str,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                **kwargs) -> pd.DataFrame:
        """
        从Alpha Vantage获取数据
        """
        try:
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                raise ValueError("请设置ALPHA_VANTAGE_API_KEY环境变量")
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params)
            data_json = response.json()
            
            if 'Time Series (Daily)' not in data_json:
                raise ValueError(f"API返回错误: {data_json}")
            
            # 转换为DataFrame
            data = pd.DataFrame.from_dict(data_json['Time Series (Daily)'], orient='index')
            data.index = pd.to_datetime(data.index)
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data = data.astype(float)
            
            # 过滤日期范围
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # 添加技术指标
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            print(f"获取Alpha Vantage数据失败: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        """
        # 计算收益率
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # 移动平均
        data['ma_5'] = data['close'].rolling(window=5).mean()
        data['ma_10'] = data['close'].rolling(window=10).mean()
        data['ma_20'] = data['close'].rolling(window=20).mean()
        data['ma_50'] = data['close'].rolling(window=50).mean()
        
        # 移动平均比率
        data['ma_ratio_5_20'] = data['ma_5'] / data['ma_20']
        data['ma_ratio_10_50'] = data['ma_10'] / data['ma_50']
        
        # 波动率
        data['volatility_5'] = data['returns'].rolling(window=5).std()
        data['volatility_20'] = data['returns'].rolling(window=20).std()
        
        # 价格位置
        data['price_position_20'] = (data['close'] - data['close'].rolling(window=20).min()) / \
                                   (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
        
        # 成交量指标
        data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_20']
        
        # 价格动量
        data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        return data
    
    def get_multiple_symbols(self,
                           symbols: List[str],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           data_source: str = 'yahoo',
                           **kwargs) -> Dict[str, pd.DataFrame]:
        """
        获取多个股票的数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据源
            **kwargs: 其他参数
            
        Returns:
            股票数据字典
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source,
                    **kwargs
                )
                data_dict[symbol] = data
                print(f"成功获取 {symbol} 的数据")
            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                continue
        
        return data_dict
    
    def preprocess_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            data: 原始数据
            **kwargs: 预处理参数
            
        Returns:
            预处理后的数据
        """
        # 处理缺失值
        data = data.ffill()
        data = data.bfill()
        
        # 删除全为NaN的行
        data = data.dropna(how='all')
        
        # 异常值处理
        if kwargs.get('remove_outliers', False):
            data = self._remove_outliers(data)
        
        # 数据标准化
        if kwargs.get('normalize', False):
            data = self._normalize_data(data)
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        移除异常值
        """
        if method == 'iqr':
            for col in data.select_dtypes(include=[np.number]).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据标准化
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['volume']:  # 跳过成交量
                continue
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val != 0:
                data[col] = (data[col] - mean_val) / std_val
        
        return data
    
    def save_data(self, data: pd.DataFrame, filepath: str, format: str = 'csv'):
        """
        保存数据
        
        Args:
            data: 数据
            filepath: 文件路径
            format: 文件格式
        """
        if format == 'csv':
            data.to_csv(filepath)
        elif format == 'parquet':
            data.to_parquet(filepath)
        elif format == 'pickle':
            data.to_pickle(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        print(f"数据已保存到: {filepath}")
    
    def load_data(self, filepath: str, format: str = 'csv') -> pd.DataFrame:
        """
        加载数据
        
        Args:
            filepath: 文件路径
            format: 文件格式
            
        Returns:
            数据DataFrame
        """
        if format == 'csv':
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == 'parquet':
            data = pd.read_parquet(filepath)
        elif format == 'pickle':
            data = pd.read_pickle(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        return data 
