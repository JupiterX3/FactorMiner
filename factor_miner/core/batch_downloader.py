#!/usr/bin/env python3
"""
智能分批下载器
为不同时间框架优化分批下载策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import ccxt
import sys
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass

from .data_downloader import DataDownloader
from .data_health_checker import health_checker
from .data_processor import data_processor


@dataclass
class BatchConfig:
    """分批下载配置"""
    timeframe: str
    batch_days: int  # 每批下载的天数
    max_candles_per_batch: int  # 每批最大K线数量
    delay_seconds: float  # 批次间延迟秒数
    retry_attempts: int  # 重试次数


class SmartBatchDownloader(DataDownloader):
    """智能分批下载器"""
    
    def __init__(self):
        super().__init__()
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
        self.logger = logging.getLogger(__name__)
        
        self._exchange_cache = {}
        
        self.batch_configs = {
            '1m': BatchConfig('1m', 1, 1000, 1.0, 3),      # 1分钟：每天一批，最多1000根K线
            '5m': BatchConfig('5m', 3, 1000, 0.8, 3),      # 5分钟：每3天一批
            '15m': BatchConfig('15m', 7, 1000, 0.6, 3),    # 15分钟：每周一批
            '30m': BatchConfig('30m', 14, 1000, 0.5, 3),   # 30分钟：每2周一批
            '1h': BatchConfig('1h', 30, 1000, 0.3, 3),     # 1小时：每月一批
            '4h': BatchConfig('4h', 90, 1000, 0.2, 3),     # 4小时：每3个月一批
            '1d': BatchConfig('1d', 365, 1000, 0.1, 3),    # 1天：每年一批
        }
    
    def get_cached_exchange(self, config_id: int = None, exchange_id: str = 'binance') -> Optional[ccxt.Exchange]:
        """获取缓存的交易所实例，避免重复调用 load_markets()"""
        cache_key = f"{exchange_id}_{config_id}_{getattr(self, 'trade_type', 'spot')}"
        
        if cache_key not in self._exchange_cache:
            exchange = self.get_exchange_instance(config_id, exchange_id)
            if exchange:
                try:
                    trade_type = getattr(self, 'trade_type', 'spot')
                    if trade_type in ['futures', 'perpetual', 'delivery']:
                        api_url = exchange.urls.get('api', {})
                        if isinstance(api_url, dict):
                            fapi_url = api_url.get('fapiPublic', 'N/A')
                            print(f"📡 使用期货API: {fapi_url}")
                        print(f"📊 加载期货市场数据 (fetchMarkets: {exchange.options.get('fetchMarkets', 'default')})")
                    else:
                        api_url = exchange.urls.get('api', {})
                        if isinstance(api_url, dict):
                            spot_url = api_url.get('public', 'N/A')
                            print(f"📡 使用现货API: {spot_url}")
                        print(f"📊 加载现货市场数据 (fetchMarkets: {exchange.options.get('fetchMarkets', 'default')})")
                    exchange.load_markets()
                    print(f"✅ 市场数据加载完成: {len(exchange.markets)} 个交易对")
                    self._exchange_cache[cache_key] = exchange
                except Exception as e:
                    self.logger.error(f"加载市场数据失败: {e}")
                    return None
            else:
                return None
        
        return self._exchange_cache.get(cache_key)
    
    def clear_exchange_cache(self):
        """清除交易所实例缓存"""
        self._exchange_cache = {}
    
    def get_batch_config(self, timeframe: str) -> BatchConfig:
        """获取时间框架的分批配置"""
        return self.batch_configs.get(timeframe, self.batch_configs['1h'])
    
    def calculate_optimal_batch_size(self, timeframe: str, start_date: str, end_date: str) -> Tuple[int, int]:
        """
        计算最优分批大小
        
        Args:
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            (每批天数, 总批次数)
        """
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end_dt - start_dt).days
        
        config = self.get_batch_config(timeframe)
        batch_days = config.batch_days
        
        # 根据总天数调整批次大小
        if total_days <= 7:
            batch_days = 1
        elif total_days <= 30:
            batch_days = 3
        elif total_days <= 90:
            batch_days = 7
        elif total_days <= 365:
            batch_days = 30
        else:
            batch_days = 90
        
        total_batches = (total_days + batch_days - 1) // batch_days
        return batch_days, total_batches
    
    def download_ohlcv_batch(self, config_id: int = None, symbol: str = None, 
                            timeframe: str = None, start_date: str = None, 
                            end_date: str = None, trade_type: str = None, 
                            progress_callback=None) -> Dict:
        """
        智能分批下载OHLCV数据
        
        Args:
            config_id: 交易所配置ID
            symbol: 交易对
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            trade_type: 交易类型
            progress_callback: 进度回调函数
            
        Returns:
            下载结果
        """
        try:
            self.trade_type = trade_type
            
            exchange = self.get_cached_exchange(config_id)
            if not exchange:
                return {'success': False, 'error': '无法创建交易所实例或加载市场数据失败'}
            
            if not exchange.markets or symbol not in exchange.markets:
                if not exchange.markets:
                    return {'success': False, 'error': '市场数据未加载，请检查网络连接或API配置'}
                available_symbols = [s for s in exchange.markets.keys() if '/USDT' in s]
                suggestions = [s for s in available_symbols if symbol.split('/')[0] in s][:5]
                return {
                    'success': False, 
                    'error': f'交易对 {symbol} 不存在',
                    'suggestions': suggestions
                }
            
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            if hasattr(self, 'trade_type') and self.trade_type:
                if self.trade_type == 'futures':
                    local_file = Path(f"data/binance/futures/{safe_symbol}-{timeframe}-futures.feather")
                elif self.trade_type == 'spot':
                    local_file = Path(f"data/binance/spot/{safe_symbol}-{timeframe}-spot.feather")
                else:
                    local_file = Path(f"data/binance/{self.trade_type}/{safe_symbol}-{timeframe}-{self.trade_type}.feather")
            else:
                local_file = Path(f"data/binance/{safe_symbol}-{timeframe}.feather")
            
            if local_file.exists():
                try:
                    existing_df = pd.read_feather(local_file)
                    time_col = None
                    for col in ['date', 'datetime', 'timestamp', 'time']:
                        if col in existing_df.columns:
                            time_col = col
                            break
                    
                    if time_col:
                        existing_df[time_col] = pd.to_datetime(existing_df[time_col], errors='coerce')
                        existing_start = existing_df[time_col].min()
                        existing_end = existing_df[time_col].max()
                        
                        if hasattr(existing_start, 'tz') and existing_start.tz is not None:
                            existing_start = existing_start.tz_localize(None)
                        if hasattr(existing_end, 'tz') and existing_end.tz is not None:
                            existing_end = existing_end.tz_localize(None)
                        
                        request_start = pd.Timestamp(start_dt)
                        request_end = pd.Timestamp(end_dt)
                        
                        if existing_start <= request_start and existing_end >= request_end:
                            if progress_callback:
                                progress_callback(100, f"✅ 本地数据已完全覆盖请求范围，跳过下载")
                            return {
                                'success': True,
                                'skipped': True,
                                'reason': '本地数据已完全覆盖请求范围',
                                'local_file': str(local_file),
                                'local_range': {
                                    'start': existing_start.strftime('%Y-%m-%d'),
                                    'end': existing_end.strftime('%Y-%m-%d')
                                },
                                'requested_range': {
                                    'start': start_date,
                                    'end': end_date
                                }
                            }
                        
                        if existing_end >= request_start and existing_end < request_end:
                            new_start_dt = existing_end + timedelta(hours=1)
                            if new_start_dt < end_dt:
                                if progress_callback:
                                    progress_callback(0, f"本地数据覆盖至 {existing_end.strftime('%Y-%m-%d')}，从 {new_start_dt.strftime('%Y-%m-%d')} 继续下载")
                                start_dt = new_start_dt
                                start_date = new_start_dt.strftime('%Y-%m-%d')
                        
                except Exception as e:
                    print(f"检查本地数据失败，继续正常下载: {e}")
            
            # 获取分批配置
            batch_days, total_batches = self.calculate_optimal_batch_size(timeframe, start_date, end_date)
            config = self.get_batch_config(timeframe)
            
            if progress_callback:
                progress_callback(0, f"开始分批下载 {symbol} {timeframe} 数据...")
                progress_callback(0, f"总批次数: {total_batches}, 每批天数: {batch_days}")
            
            # 分批下载数据
            all_data = []
            current_dt = start_dt
            batch_count = 0
            
            while current_dt < end_dt:
                try:
                    batch_count += 1
                    
                    # 计算本次下载的结束时间
                    batch_end = min(current_dt + timedelta(days=batch_days), end_dt)
                    
                    # 计算进度（按秒计算，避免时间范围不足 1 天时出现除零）
                    total_seconds = (end_dt - start_dt).total_seconds()
                    elapsed_seconds = (current_dt - start_dt).total_seconds()
                    if total_seconds <= 0:
                        progress = 0
                    else:
                        progress = min(95, int(max(0, elapsed_seconds) / total_seconds * 90))
                    
                    if progress_callback:
                        progress_callback(progress, f"下载第 {batch_count}/{total_batches} 批: "
                                        f"{current_dt.strftime('%Y-%m-%d')} 到 {batch_end.strftime('%Y-%m-%d')}")
                    
                    # 下载数据（期货口径保留 taker_buy_base/quote）
                    ohlcv = self._fetch_klines_with_taker(
                        exchange,
                        symbol,
                        timeframe,
                        int(current_dt.timestamp() * 1000),
                        limit=config.max_candles_per_batch,
                    )
                    
                    if ohlcv:
                        all_data.extend(ohlcv)
                        if progress_callback:
                            progress_callback(progress + 5, f"第 {batch_count} 批完成，"
                                            f"当前总计: {len(all_data)} 条数据")
                    else:
                        self.logger.warning(f"第 {batch_count} 批没有数据")
                    
                    # 移动到下一批
                    current_dt = batch_end
                    
                    # 限速和延迟
                    time.sleep(config.delay_seconds)
                    
                    # 检查是否需要重试
                    if batch_count % 10 == 0:  # 每10批检查一次
                        time.sleep(config.delay_seconds * 2)  # 额外延迟
                    
                except Exception as e:
                    self.logger.error(f"第 {batch_count} 批下载失败: {e}")
                    
                    # 重试逻辑
                    retry_count = 0
                    while retry_count < config.retry_attempts:
                        try:
                            time.sleep(config.delay_seconds * 2)
                            retry_count += 1
                            
                            if progress_callback:
                                progress_callback(progress, f"第 {batch_count} 批重试 {retry_count}/{config.retry_attempts}")
                            
                            ohlcv = self._fetch_klines_with_taker(
                                exchange,
                                symbol,
                                timeframe,
                                int(current_dt.timestamp() * 1000),
                                limit=config.max_candles_per_batch,
                            )
                            
                            if ohlcv:
                                all_data.extend(ohlcv)
                                break
                            else:
                                self.logger.warning(f"第 {batch_count} 批重试 {retry_count} 次后仍无数据")
                                
                        except Exception as retry_e:
                            self.logger.error(f"第 {batch_count} 批重试 {retry_count} 失败: {retry_e}")
                    
                    # 如果重试失败，继续下一批
                    current_dt = batch_end
            
            if not all_data:
                return {'success': False, 'error': '没有下载到数据'}
            
            if progress_callback:
                progress_callback(95, f"数据下载完成，共 {len(all_data)} 条，正在处理...")
            
            # 转换为DataFrame - 直接命名为 date，避免后续复杂操作
            # 期货口径下 all_data 每行 8 列（含 taker_buy_base/quote），
            # 其余情况 taker 两列为 None，保持列对齐
            df = pd.DataFrame(
                all_data,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'taker_buy_base', 'taker_buy_quote',
                ],
            )
            # 转换时间戳并处理时区问题 - 直接命名为 date
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 如果时间戳是UTC时间，转换为本地时间
            # 注意：CCXT返回的时间戳通常是UTC时间
            if df['date'].dt.tz is None:
                # 假设是UTC时间，转换为本地时间
                df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            
            df.set_index('date', inplace=True)  # 设置 date 为索引
            df.drop('timestamp', axis=1, inplace=True)
            
            print(f"原始下载数据: {len(df)} 条")
            print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
            
            # 去重和排序（按 date 索引）
            df = df[~df.index.duplicated(keep='last')].sort_index()
            print(f"去重后数据: {len(df)} 条")
            
            # 阶段1: 与现有数据合并和验证
            df_merged = self._merge_with_existing_data(df, symbol, timeframe, start_date, end_date)
            
            # 阶段2: 检测和补全数据间断
            df_complete = self._fill_data_gaps(df_merged, symbol, timeframe, start_date, end_date, progress_callback)
            
            # 阶段3: 最终验证
            validation_passed = self._final_validation(df_complete, timeframe, symbol)
            validation_warning = None
            
            if not validation_passed:
                print("⚠️ 数据验证未达100分，开始自动修复...")
                df_complete = self._auto_fix_data_issues(df_complete, timeframe, symbol, max_retries=3)
                
                validation_passed = self._final_validation(df_complete, timeframe, symbol)
                if not validation_passed:
                    gaps = self._detect_data_gaps(df_complete, timeframe)
                    coverage = self._calculate_coverage(df_complete, timeframe)
                    validation_warning = f"数据未达100分标准(覆盖率{coverage:.1f}%, {len(gaps)}个间断)，但仍已保存"
                    print(f"⚠️ {validation_warning}")
            
            # 准备保存数据
            print("=== 保存前数据检查 ===")
            print(f"df_complete 索引名: {df_complete.index.name}")
            print(f"df_complete 列名: {df_complete.columns.tolist()}")
            print(f"df_complete 形状: {df_complete.shape}")
            print(f"df_complete 数据类型:")
            print(df_complete.dtypes)
            print("========================")
            
            # 检查是否需要重置索引
            # if df_complete.index.name == 'date' and 'date' in df_complete.columns:
            #     print("⚠️ 检测到 date 索引和 date 列冲突，需要重置索引")
            #     df_save = df_complete.reset_index()
            # else:
            #     print("✅ 没有索引和列冲突，直接使用原数据")
            #     df_save = df_complete.copy()
            df_save = df_complete.copy()

            print(f"准备保存: {len(df_save)} 条数据")
            
            if progress_callback:
                msg = f"数据处理完成，准备保存..."
                if validation_warning:
                    msg = f"数据处理完成(⚠️{validation_warning})，准备保存..."
                progress_callback(98, msg)
            
            # 保存数据
            save_result = self._save_data_with_merge(df_save, symbol, timeframe, start_date, end_date)
            
            if progress_callback:
                progress_callback(100, f"下载完成！{save_result.get('message', '数据已保存')}")
            
            result = {
                'success': True,
                'data': df_save,
                'total_records': len(df_save),
                'timeframe': timeframe,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'batch_info': {
                    'total_batches': total_batches,
                    'batch_days': batch_days,
                    'actual_batches': batch_count
                },
                'message': save_result.get('message', '数据下载完成')
            }
            if validation_warning:
                result['validation_warning'] = validation_warning
            return result
            
        except Exception as e:
            error_msg = f"下载失败: {e}"
            self.logger.error(error_msg)
            if progress_callback:
                progress_callback(0, error_msg)
            return {'success': False, 'error': error_msg}
    
    def _save_data_with_merge(self, df_save: pd.DataFrame, symbol: str, timeframe: str, 
                             start_date: str, end_date: str) -> Dict:
        """保存数据并处理合并逻辑"""
        # 直接保存数据，避免重复调用
        try:
            safe_symbol = symbol.replace('/', '_').replace(':', '_')
            
            if hasattr(self, 'trade_type') and self.trade_type:
                if self.trade_type == 'futures':
                    filename = f"{safe_symbol}-{timeframe}-futures.feather"
                elif self.trade_type == 'spot':
                    filename = f"{safe_symbol}-{timeframe}-spot.feather"
                elif self.trade_type in ['perpetual', 'delivery']:
                    filename = f"{safe_symbol}-{timeframe}-{self.trade_type}.feather"
                else:
                    filename = f"{safe_symbol}_{timeframe}_{start_date}_{end_date}.feather"
            else:
                filename = f"{safe_symbol}_{timeframe}_{start_date}_{end_date}.feather"
            print(f"完成构建文件名: {filename}")

            # 确定存储目录
            if hasattr(self, 'trade_type') and self.trade_type:
                if self.trade_type == 'futures':
                    save_path = Path("data/binance/futures") / filename
                elif self.trade_type == 'spot':
                    save_path = Path("data/binance/spot") / filename
                elif self.trade_type in ['perpetual', 'delivery']:
                    save_path = Path(f"data/binance/{self.trade_type}") / filename
                else:
                    save_path = Path("data/binance") / filename
            else:
                save_path = Path("data/binance") / filename
            print(f"生成保存路径: {save_path}")
            
            # 检查现有文件并合并
            if save_path.exists():
                try:
                    existing_df = pd.read_feather(save_path)
                    print(f"读取到现有数据: {len(existing_df)} 条")
                    print(f"现有数据列名: {existing_df.columns.tolist()}")
                    
                    # 清理现有数据中的垃圾列
                    cols_to_drop = ['level_0', 'index']
                    for col in cols_to_drop:
                        if col in existing_df.columns:
                            existing_df = existing_df.drop(columns=[col])
                            print(f"删除垃圾列: {col}")
                    
                    # 确保现有数据有 date 列
                    if 'date' not in existing_df.columns:
                        if existing_df.index.name == 'date':
                            existing_df = existing_df.reset_index()
                            print("从索引恢复 date 列")
                        else:
                            print("❌ 现有数据没有 date 列，跳过合并")
                            existing_df = None
                    
                    if existing_df is not None and len(existing_df) > 0:
                        # 转换 date 列为 datetime 类型
                        try:
                            if existing_df['date'].dtype in ['int64', 'int32']:
                                sample_date = existing_df['date'].dropna().iloc[0] if len(existing_df['date'].dropna()) > 0 else 0
                                if sample_date > 1e12:
                                    existing_df['date'] = pd.to_datetime(existing_df['date'], unit='ms')
                                else:
                                    existing_df['date'] = pd.to_datetime(existing_df['date'], unit='s')
                            elif existing_df['date'].dtype == 'object':
                                existing_df['date'] = pd.to_datetime(existing_df['date'])
                            print(f"✅ 现有数据 date 列类型: {existing_df['date'].dtype}")
                        except Exception as e:
                            print(f"❌ 转换现有数据 date 列失败: {e}")
                            existing_df = None
                    
                    # 处理新数据：确保有 date 列
                    df_new = df_save.copy()
                    if df_new.index.name == 'date':
                        if 'date' in df_new.columns:
                            df_new = df_new.reset_index(drop=True)
                            print("新数据索引为 date 且已存在 date 列，已丢弃索引避免重复列")
                        else:
                            df_new = df_new.reset_index()
                            print("新数据从索引恢复 date 列")
                    
                    if 'date' not in df_new.columns:
                        print("❌ 新数据没有 date 列，无法合并")
                    elif existing_df is not None and len(existing_df) > 0:
                        # 过滤掉无效日期
                        existing_df = existing_df[existing_df['date'].notna()]
                        df_new = df_new[df_new['date'].notna()]
                        
                        # 统一时区处理：将所有数据转为 tz-naive
                        if hasattr(existing_df['date'].dtype, 'tz') and existing_df['date'].dtype.tz is not None:
                            existing_df['date'] = existing_df['date'].dt.tz_localize(None)
                            print("现有数据时区已移除")
                        if hasattr(df_new['date'].dtype, 'tz') and df_new['date'].dtype.tz is not None:
                            df_new['date'] = df_new['date'].dt.tz_localize(None)
                            print("新数据时区已移除")
                        
                        print(f"合并前：现有数据 {len(existing_df)} 条，新数据 {len(df_new)} 条")
                        
                        # 合并数据
                        combined_df = pd.concat([existing_df, df_new], ignore_index=True)
                        
                        # 按 date 去重，保留最新的数据
                        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
                        combined_df = combined_df.sort_values('date').reset_index(drop=True)
                        
                        print(f"合并完成：原数据 {len(existing_df)} 行，新数据 {len(df_new)} 行，合并后 {len(combined_df)} 行")
                        df_save = combined_df
                    
                except Exception as e:
                    print(f"⚠️ 合并数据失败（{e}），将用新数据直接覆盖旧文件")
                    import traceback
                    traceback.print_exc()
                    df_save = df_new.copy()
            
            # 确保保存前数据格式正确
            if df_save.index.name == 'date':
                if 'date' in df_save.columns:
                    df_save = df_save.reset_index(drop=True)
                else:
                    df_save = df_save.reset_index()
            
            # 过滤无效日期
            if 'date' in df_save.columns:
                df_save = df_save[df_save['date'].notna()]
            
            # 删除可能存在的垃圾列
            cols_to_drop = ['level_0', 'index']
            for col in cols_to_drop:
                if col in df_save.columns:
                    df_save = df_save.drop(columns=[col])
            
            # 保存数据
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_save.to_feather(save_path)
            print(f"保存数据成功，共 {len(df_save)} 条记录")

            return {
                'success': True,
                'message': f'数据保存成功，共 {len(df_save)} 条记录',
                'file_path': str(save_path)
            }
            
        except Exception as e:
            print(f"保存数据失败: {e}")
            return {
                'success': False,
                'error': f'保存失败: {e}'
            }


    def _fix_data_issues(self, df: pd.DataFrame, health_report: Dict) -> pd.DataFrame:
        """
        修复数据问题
        
        Args:
            df: 原始数据
            health_report: 健康度检查报告
            
        Returns:
            修复后的数据
        """
        try:
            df_fixed = df.copy()
            issues = health_report.get('issues', [])
            
            # 首先检查数据类型问题
            df_fixed = self._fix_data_types(df_fixed)
            
            for issue in issues:
                if 'OHLC数据逻辑错误' in issue:
                    # 修复OHLC逻辑错误
                    df_fixed = self._fix_ohlc_logic(df_fixed)
                elif '价格为0或负数' in issue:
                    # 修复价格问题
                    df_fixed = self._fix_price_issues(df_fixed)
                elif '成交量为负数' in issue:
                    # 修复成交量问题
                    df_fixed = self._fix_volume_issues(df_fixed)
            
            # 最终去重 - 修复：使用正确的列名
            if 'datetime' in df_fixed.columns:
                df_fixed = data_processor.remove_duplicates(df_fixed, 'datetime')
            elif 'date' in df_fixed.columns:
                df_fixed = data_processor.remove_duplicates(df_fixed, 'date')
            else:
                print("⚠️  警告：找不到时间列，跳过去重")
            
            print(f"🔧 数据修复完成，原始数据 {len(df)} 条，修复后 {len(df_fixed)} 条")
            return df_fixed
            
        except Exception as e:
            print(f"❌ 数据修复失败: {e}")
            return df
    
    def _fix_ohlc_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复OHLC逻辑错误"""
        try:
            df_fixed = df.copy()
            
            # 确保 high >= low
            df_fixed['high'] = df_fixed[['high', 'low']].max(axis=1)
            df_fixed['low'] = df_fixed[['high', 'low']].min(axis=1)
            
            # 确保 open 和 close 在 high 和 low 之间
            df_fixed['open'] = df_fixed['open'].clip(df_fixed['low'], df_fixed['high'])
            df_fixed['close'] = df_fixed['close'].clip(df_fixed['low'], df_fixed['high'])
            
            return df_fixed
        except Exception as e:
            print(f"❌ 修复OHLC逻辑失败: {e}")
            return df
    
    def _fix_price_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复价格问题"""
        try:
            df_fixed = df.copy()
            
            # 将0或负数价格替换为前一个有效价格
            for col in ['open', 'high', 'low', 'close']:
                if col in df_fixed.columns:
                    df_fixed[col] = df_fixed[col].replace([0, -np.inf, np.inf], np.nan)
                    df_fixed[col] = df_fixed[col].ffill()
            
            return df_fixed
        except Exception as e:
            print(f"❌ 修复价格问题失败: {e}")
            return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复数据类型问题"""
        try:
            df_fixed = df.copy()
            
            # 修复OHLC列的数据类型
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                if col in df_fixed.columns:
                    # 如果列是datetime类型但应该是数值，进行修复
                    if pd.api.types.is_datetime64_any_dtype(df_fixed[col]):
                        print(f"⚠️  发现 {col} 列类型错误（应该是数值但实际是datetime），尝试修复...")
                        try:
                            # 尝试转换为数值类型
                            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
                            # 如果转换失败，用前一个有效值填充
                            if df_fixed[col].isna().all():
                                print(f"❌ 无法修复 {col} 列，将使用前一个有效值")
                                df_fixed[col] = df_fixed[col].ffill()
                        except Exception as e:
                            print(f"❌ 修复 {col} 列失败: {e}")
                            # 使用前一个有效值填充
                            df_fixed[col] = df_fixed[col].ffill()
            
            return df_fixed
            
        except Exception as e:
            print(f"❌ 修复数据类型失败: {e}")
            return df
    
    def _fix_volume_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复成交量问题"""
        try:
            df_fixed = df.copy()
            
            if 'volume' in df_fixed.columns:
                # 将负数成交量替换为0
                df_fixed['volume'] = df_fixed['volume'].clip(lower=0)
            
            return df_fixed
        except Exception as e:
            print(f"❌ 修复成交量问题失败: {e}")
            return df


# 创建全局实例
batch_downloader = SmartBatchDownloader()
