"""
因子引擎
基于透明JSON存储的因子计算引擎
"""

import pandas as pd
import numpy as np
import logging
import threading
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from factor_miner.core.factor_storage import TransparentFactorStorage, get_global_storage

logger = logging.getLogger(__name__)

_module_cache: Dict[str, Any] = {}
_module_cache_lock = threading.Lock()
_sys_path_dirs: set = set()
_sys_path_lock = threading.Lock()


def _ensure_sys_path(dir_path: str) -> None:
    """线程安全地向 sys.path 添加目录（去重）"""
    import sys
    with _sys_path_lock:
        if dir_path not in _sys_path_dirs:
            if dir_path not in sys.path:
                sys.path.insert(0, dir_path)
            _sys_path_dirs.add(dir_path)


def _load_module_cached(cache_key: str, func_path, module_name: str):
    """
    线程安全的模块加载，带文件修改时间缓存失效。
    
    Returns:
        (module, error_msg) - module 为加载的模块，error_msg 为错误信息
    """
    import importlib.util
    import os

    mtime = os.path.getmtime(func_path) if os.path.exists(func_path) else None

    with _module_cache_lock:
        entry = _module_cache.get(cache_key)
        if entry is not None:
            cached_mtime, module = entry
            if mtime is not None and cached_mtime == mtime:
                return module, None

    spec = importlib.util.spec_from_file_location(module_name, func_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, str(e)

    with _module_cache_lock:
        _module_cache[cache_key] = (mtime, module)

    return module, None


class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self, storage: TransparentFactorStorage = None):
        """
        初始化因子引擎
        
        Args:
            storage: 因子存储实例，如果为None则使用全局实例
        """
        self.storage = storage or get_global_storage()
        logger.info("因子引擎已初始化 (V3 扁平化目录)")
    
    def compute_single_factor(self, factor_id: str, data: pd.DataFrame, **kwargs) -> Optional[pd.Series]:
        """
        计算单个因子

        v4 支持三种因子类型：
        1. function - 函数类型因子，使用 function_file 和 entry_point
        2. formula  - 纯公式类型因子
        3. ml_model - 算法代理类型因子：由 user_algo/<algorithm_name>.py 的
                      `calculate_single_factor(data, factor_name, ...)` 执行。
                      典型用途是 GP 截面挖掘因子（gp_cross_sectional 等）。
                      真正依赖 pkl 模型推理的 ML 预训练因子已在 v4 中移除。

        Args:
            factor_id: 因子ID
            data: OHLCV 及可选的 extra 列（由 DataLoader.load_with_extras 注入）
            **kwargs: 覆盖默认参数
        """
        try:
            logger.debug(f"开始计算因子: {factor_id}")

            factor_def = self.storage.load_factor_definition(factor_id)
            if not factor_def:
                logger.error(f"找不到因子定义: {factor_id}")
                return None

            computation_type = factor_def.computation_type

            if computation_type == "function":
                return self._compute_function_factor(factor_def, data, kwargs)
            elif computation_type == "formula":
                return self._compute_formula_factor(factor_def, data, kwargs)
            elif computation_type == "ml_model":
                return self._compute_algorithm_proxy_factor(factor_def, data, kwargs)
            else:
                logger.error(f"不支持的计算类型: {computation_type}")
                return None

        except Exception as e:
            logger.error(f"计算因子失败 {factor_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_function_factor(self, factor_def, data: pd.DataFrame, kwargs: dict) -> Optional[pd.Series]:
        """计算函数类型因子"""
        comp_data = factor_def.computation_data or {}
        
        function_file = comp_data.get('function_file')
        if not function_file:
            logger.error(f"因子定义中缺少 function_file: {factor_def.factor_id}")
            return None
        
        # function_file 通常是相对 storage_dir 的路径（如 basic_kline/functions/xxx.py）；
        # 兼容老数据（只有裸文件名 xxx.py 的情况）：扫描所有一级目录的 functions/
        possible_paths = [self.storage.storage_dir / function_file]
        # 老版 V3 路径兼容
        for legacy in ("technicals", "minactors"):
            possible_paths.append(self.storage.storage_dir / legacy / function_file)
        fname = Path(function_file).name
        for group in self.storage.list_source_groups():
            possible_paths.append(
                self.storage.storage_dir / group / "functions" / fname
            )

        func_path = None
        for path in possible_paths:
            if path.exists():
                func_path = path
                break

        if not func_path:
            logger.error(f"函数文件不存在，尝试的路径: {[str(p) for p in possible_paths]}")
            return None
        
        factorlib_path = str(self.storage.storage_dir.parent)
        _ensure_sys_path(factorlib_path)
        # 允许因子文件 `from _alpha_ops import ...`（qlib158/qlib360/wq101 共享算子）
        _ensure_sys_path(str(func_path.parent))
        
        cache_key = str(func_path)
        module, err = _load_module_cached(cache_key, func_path, f"factor_{factor_def.factor_id}")
        if err:
            logger.error(f"加载函数模块失败: {err}")
            return None
        
        entry_point = comp_data.get('entry_point', 'calculate')
        if not hasattr(module, entry_point):
            logger.error(f"函数中未找到入口点: {entry_point}")
            return None
        
        func = getattr(module, entry_point)
        
        params = (factor_def.parameters or {}).copy()
        params.update(kwargs)
        
        try:
            result = func(data, **params)
            if result is not None:
                logger.debug(f"因子计算成功: {factor_def.factor_id}")
                return result
            else:
                logger.warning(f"因子计算返回空结果: {factor_def.factor_id}")
                return None
        except Exception as e:
            logger.error(f"调用因子函数失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_formula_factor(self, factor_def, data: pd.DataFrame, kwargs: dict) -> Optional[pd.Series]:
        """计算公式类型因子"""
        comp_data = factor_def.computation_data or {}
        formula = comp_data.get('formula', '')
        
        if not formula:
            logger.error(f"因子定义中缺少 formula: {factor_def.factor_id}")
            return None
        
        if formula.strip().startswith('#'):
            logger.info(f"公式为注释，尝试使用函数文件: {factor_def.factor_id}")
            return self._compute_formula_as_function(factor_def, data, kwargs)
        
        params = (factor_def.parameters or {}).copy()
        params.update(kwargs)
        
        try:
            local_vars = {
                'close': data['close'],
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'volume': data['volume'],
                'data': data,
                'pd': pd,
                'np': np,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'round': round,
                'float': float,
                'int': int,
                'str': str,
            }
            
            local_vars.update(params)
            
            result = eval(formula, {"__builtins__": {}}, local_vars)
            
            if isinstance(result, pd.Series):
                logger.debug(f"公式因子计算成功: {factor_def.factor_id}")
                return result
            else:
                logger.warning(f"公式因子返回非Series类型: {type(result)}")
                return None
                
        except Exception as e:
            logger.error(f"公式因子计算失败 {factor_def.factor_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_formula_as_function(self, factor_def, data: pd.DataFrame, kwargs: dict) -> Optional[pd.Series]:
        """将公式类型因子作为函数类型处理"""
        factor_id = factor_def.factor_id
        
        base_name = factor_id.split('_')[0]
        name_without_suffix = '_'.join(factor_id.split('_')[:-1]) if '_' in factor_id else factor_id
        
        possible_paths = []
        for group in self.storage.list_source_groups():
            possible_paths.append(
                self.storage.storage_dir / group / "functions" / f"{factor_id}.py"
            )
            possible_paths.append(
                self.storage.storage_dir / group / "functions" / f"{name_without_suffix}.py"
            )
            possible_paths.append(
                self.storage.storage_dir / group / "functions" / f"{base_name}.py"
            )
        possible_paths.extend([
            self.storage.storage_dir / "technicals" / "functions" / f"{factor_id}.py",
            self.storage.storage_dir / "minactors" / "functions" / f"{factor_id}.py",
            self.storage.storage_dir / "technicals" / "functions" / f"{name_without_suffix}.py",
            self.storage.storage_dir / "technicals" / "functions" / f"{base_name}.py",
        ])
        
        func_path = None
        for path in possible_paths:
            if path.exists():
                func_path = path
                break
        
        if not func_path:
            comp_data = factor_def.computation_data or {}
            formula = comp_data.get('formula', '')
            if formula and not formula.strip().startswith('#'):
                return self._execute_formula(formula, data, factor_def.parameters or {}, kwargs)
            
            logger.warning(f"找不到函数文件且公式为空/注释，跳过因子: {factor_id}")
            return None
        
        factorlib_path = str(self.storage.storage_dir.parent)
        _ensure_sys_path(factorlib_path)
        
        cache_key = str(func_path)
        module, err = _load_module_cached(cache_key, func_path, f"factor_{factor_id}")
        if err:
            logger.error(f"加载函数模块失败: {err}")
            return None
        
        if not hasattr(module, 'calculate'):
            logger.error(f"函数中未找到 calculate 入口点")
            return None
        
        func = getattr(module, 'calculate')
        
        params = (factor_def.parameters or {}).copy()
        params.update(kwargs)
        
        try:
            result = func(data, **params)
            if result is not None:
                logger.debug(f"公式因子(函数模式)计算成功: {factor_id}")
                return result
            else:
                logger.warning(f"公式因子(函数模式)返回空结果: {factor_id}")
                return None
        except Exception as e:
            logger.error(f"调用因子函数失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _execute_formula(self, formula: str, data: pd.DataFrame, 
                         factor_params: dict, kwargs: dict) -> Optional[pd.Series]:
        """执行公式计算"""
        import pandas as pd
        import numpy as np
        
        params = factor_params.copy()
        params.update(kwargs)
        
        local_vars = {
            'data': data,
            'pd': pd,
            'np': np,
            **params
        }
        
        try:
            exec(formula, local_vars)
            if 'result' in local_vars:
                return local_vars['result']
            else:
                logger.error("公式执行后未找到 'result' 变量")
                return None
        except Exception as e:
            logger.error(f"公式执行失败: {e}")
            return None
    
    def _compute_algorithm_proxy_factor(self, factor_def, data: pd.DataFrame, kwargs: dict) -> Optional[pd.Series]:
        """
        算法代理因子计算（computation_type=ml_model）。

        通过 `algorithm_name` 指向 `user_algo/<algorithm_name>.py`，调用其
        `calculate_single_factor(data, factor_name, factor_id=..., computation_data=..., **kwargs)`
        完成计算。主要用于 GP 挖掘因子一类"批量生成共享算子"的场景。
        """
        comp_data = factor_def.computation_data or {}
        algorithm_name = comp_data.get('algorithm_name')
        if not algorithm_name:
            logger.error(
                f"因子 {factor_def.factor_id} 是 ml_model(算法代理) 但缺少 algorithm_name"
            )
            return None

        algorithm_module = self._load_algorithm_module(algorithm_name)
        if algorithm_module is None:
            return None

        if not hasattr(algorithm_module, 'calculate_single_factor'):
            logger.error(
                f"算法模块 {algorithm_name} 缺少 calculate_single_factor 函数"
            )
            return None

        return algorithm_module.calculate_single_factor(
            data,
            factor_def.name,
            factor_id=factor_def.factor_id,
            computation_data=comp_data,
            **kwargs,
        )
    
    def _load_algorithm_module(self, algorithm_name: str):
        """动态加载算法模块（带缓存和并发安全）"""
        try:
            from pathlib import Path
            
            algo_dir = Path(__file__).parent.parent.parent / "user_algo"
            _ensure_sys_path(str(algo_dir))
            
            algo_file = algo_dir / f"{algorithm_name}.py"
            if not algo_file.exists():
                logger.error(f"算法文件不存在: {algo_file}")
                return None
            
            cache_key = str(algo_file)
            module, err = _load_module_cached(cache_key, algo_file, algorithm_name)
            if err:
                logger.error(f"加载算法模块失败 {algorithm_name}: {err}")
                return None
            
            return module
            
        except Exception as e:
            logger.error(f"加载算法模块失败 {algorithm_name}: {e}")
            return None
    
    def compute_multiple_factors(self, factor_ids: List[str], data: pd.DataFrame, 
                                **kwargs) -> pd.DataFrame:
        """
        批量计算多个因子
        
        Args:
            factor_ids: 因子ID列表
            data: OHLCV数据
            **kwargs: 公共参数
            
        Returns:
            DataFrame，每列一个因子
        """
        results = {}
        errors = []
        
        for factor_id in factor_ids:
            try:
                result = self.compute_single_factor(factor_id, data, **kwargs)
                if result is not None:
                    results[factor_id] = result
                else:
                    errors.append(f"{factor_id}: 计算返回空结果")
            except Exception as e:
                errors.append(f"{factor_id}: {str(e)}")
                logger.error(f"批量计算中因子失败 {factor_id}: {e}")
        
        if errors:
            logger.warning(f"批量计算中的错误: {errors}")
        
        if not results:
            logger.warning("批量计算无任何成功结果")
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def compute_factor_category(self, category: str, data: pd.DataFrame, 
                               **kwargs) -> pd.DataFrame:
        """
        按分类批量计算因子
        
        Args:
            category: 因子分类
            data: OHLCV数据
            **kwargs: 公共参数
            
        Returns:
            DataFrame，每列一个因子
        """
        factor_ids = self.storage.get_factors_by_category(category)
        
        if not factor_ids:
            logger.warning(f"分类 {category} 下没有找到因子")
            return pd.DataFrame()
        
        logger.info(f"分类 {category} 下找到 {len(factor_ids)} 个因子")
        return self.compute_multiple_factors(factor_ids, data, **kwargs)
    
    def list_factors(self) -> List[str]:
        """获取所有可用因子列表"""
        return self.storage.list_factors()
    
    def list_categories(self) -> List[str]:
        """获取所有分类列表"""
        categories = set()
        for factor_id in self.list_factors():
            factor_def = self.storage.load_factor_definition(factor_id)
            if factor_def:
                categories.add(factor_def.category)
        return sorted(list(categories))
    
    def get_factor_info(self, factor_id: str) -> Optional[Dict]:
        """
        获取因子详细信息
        
        Args:
            factor_id: 因子ID
            
        Returns:
            因子信息字典
        """
        factor_def = self.storage.load_factor_definition(factor_id)
        if factor_def:
            return factor_def.to_dict()
        return None
    
    def search_factors(self, query: str = "", category: str = "", 
                      computation_type: str = "") -> List[Dict]:
        """
        搜索因子
        
        Args:
            query: 搜索关键词（匹配名称或描述）
            category: 按分类过滤
            computation_type: 按计算类型过滤
            
        Returns:
            匹配的因子信息列表
        """
        results = []
        
        for factor_id in self.list_factors():
            factor_def = self.storage.load_factor_definition(factor_id)
            if not factor_def:
                continue
            
            # 分类过滤
            if category and factor_def.category != category:
                continue
            
            # 计算类型过滤
            if computation_type and factor_def.computation_type != computation_type:
                continue
            
            # 关键词搜索
            if query:
                text = f"{factor_def.name} {factor_def.description} {factor_def.factor_id}".lower()
                if query.lower() not in text:
                    continue
            
            results.append(factor_def.to_dict())
        
        return results
    
    def validate_factor(self, factor_id: str, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        验证因子配置
        
        Args:
            factor_id: 因子ID
            data: 测试数据（可选）
            
        Returns:
            验证结果
        """
        result = {
            'factor_id': factor_id,
            'exists': False,
            'valid_definition': False,
            'computable': False,
            'errors': []
        }
        
        try:
            # 检查因子是否存在
            factor_def = self.storage.load_factor_definition(factor_id)
            if not factor_def:
                result['errors'].append("因子定义不存在")
                return result
            
            result['exists'] = True
            result['definition'] = factor_def.to_dict()
            
            # 验证定义完整性
            required_fields = ['factor_id', 'name', 'computation_type', 'computation_data']
            for field in required_fields:
                if not hasattr(factor_def, field) or getattr(factor_def, field) is None:
                    result['errors'].append(f"缺少必需字段: {field}")
            
            if not result['errors']:
                result['valid_definition'] = True
            
            # 如果提供了数据，测试计算
            if data is not None and result['valid_definition']:
                try:
                    test_result = self.compute_single_factor(factor_id, data)
                    if test_result is not None:
                        result['computable'] = True
                        result['test_result_shape'] = test_result.shape
                        result['test_result_sample'] = test_result.head().to_dict()
                    else:
                        result['errors'].append("计算返回空结果")
                except Exception as e:
                    result['errors'].append(f"计算测试失败: {str(e)}")
            
        except Exception as e:
            result['errors'].append(f"验证过程出错: {str(e)}")
        
        return result
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        factors = self.list_factors()
        categories = self.list_categories()
        
        # 按分类统计
        category_stats = {}
        computation_type_stats = {}
        
        for factor_id in factors:
            factor_def = self.storage.load_factor_definition(factor_id)
            if factor_def:
                # 分类统计
                cat = factor_def.category
                if cat not in category_stats:
                    category_stats[cat] = 0
                category_stats[cat] += 1
                
                # 计算类型统计
                comp_type = factor_def.computation_type
                if comp_type not in computation_type_stats:
                    computation_type_stats[comp_type] = 0
                computation_type_stats[comp_type] += 1
        
        return {
            'total_factors': len(factors),
            'total_categories': len(categories),
            'categories': categories,
            'category_stats': category_stats,
            'computation_type_stats': computation_type_stats,
            'storage_path': str(self.storage.storage_dir)
        }
    
    def export_factor_list(self, output_file: str = None) -> Dict[str, Any]:
        """
        导出因子列表
        
        Args:
            output_file: 输出文件路径（可选）
            
        Returns:
            因子列表数据
        """
        factors_data = []
        
        for factor_id in self.list_factors():
            factor_def = self.storage.load_factor_definition(factor_id)
            if factor_def:
                factors_data.append({
                    'factor_id': factor_def.factor_id,
                    'name': factor_def.name,
                    'description': factor_def.description,
                    'category': factor_def.category,
                    'computation_type': factor_def.computation_type,
                    'parameters': factor_def.parameters,
                    'created_at': factor_def.metadata.get('created_at')
                })
        
        export_data = {
            'export_time': pd.Timestamp.now().isoformat(),
            'total_factors': len(factors_data),
            'factors': factors_data
        }
        
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            logger.info(f"因子列表已导出到: {output_file}")
        
        return export_data


# 全局实例
_global_engine = None

def get_global_engine() -> FactorEngine:
    """获取全局引擎实例"""
    global _global_engine
    if _global_engine is None:
        _global_engine = FactorEngine()
    return _global_engine