"""
因子引擎（兼容包装层）

核心执行逻辑已迁移到 FactorExecutor，本模块仅保留兼容接口。
新代码应直接使用 FactorExecutor。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from factor_miner.core.factor_storage import TransparentFactorStorage, get_global_storage

logger = logging.getLogger(__name__)

import warnings


class FactorEngine:
    """因子计算引擎（兼容包装层，核心执行已迁移到 FactorExecutor）"""

    def __init__(self, storage: TransparentFactorStorage = None):
        self.storage = storage or get_global_storage()
        self._executor = None
        logger.info("因子引擎已初始化 (V3 扁平化目录)")

    def _get_executor(self):
        if self._executor is None:
            from .factor_executor import FactorExecutor
            self._executor = FactorExecutor()
        return self._executor

    def compute_single_factor(self, factor_id: str, data: pd.DataFrame, **kwargs) -> Optional[pd.Series]:
        try:
            return self._get_executor().compute(factor_id, data, **kwargs)
        except ValueError:
            return None
        except Exception as e:
            logger.error(f"计算因子失败 {factor_id}: {e}")
            return None

    def compute_multiple_factors(self, factor_ids: List[str], data: pd.DataFrame,
                                **kwargs) -> pd.DataFrame:
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
        factor_ids = self.storage.get_factors_by_category(category)

        if not factor_ids:
            logger.warning(f"分类 {category} 下没有找到因子")
            return pd.DataFrame()

        logger.info(f"分类 {category} 下找到 {len(factor_ids)} 个因子")
        return self.compute_multiple_factors(factor_ids, data, **kwargs)

    def list_factors(self) -> List[str]:
        warnings.warn(
            "FactorEngine.list_factors() is deprecated, use FactorCatalogService.list_factors()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.storage.list_factors()

    def list_categories(self) -> List[str]:
        warnings.warn(
            "FactorEngine.list_categories() is deprecated, use FactorCatalogService",
            DeprecationWarning,
            stacklevel=2,
        )
        categories = set()
        for factor_id in self.list_factors():
            factor_def = self.storage.load_factor_definition(factor_id)
            if factor_def:
                categories.add(factor_def.category)
        return sorted(list(categories))

    def get_factor_info(self, factor_id: str) -> Optional[Dict]:
        warnings.warn(
            "FactorEngine.get_factor_info() is deprecated, use FactorCatalogService or FactorRepository",
            DeprecationWarning,
            stacklevel=2,
        )
        factor_def = self.storage.load_factor_definition(factor_id)
        if factor_def:
            return factor_def.to_dict()
        return None

    def search_factors(self, query: str = "", category: str = "",
                      computation_type: str = "") -> List[Dict]:
        warnings.warn(
            "FactorEngine.search_factors() is deprecated, use FactorCatalogService.query_factors()",
            DeprecationWarning,
            stacklevel=2,
        )
        results = []

        for factor_id in self.list_factors():
            factor_def = self.storage.load_factor_definition(factor_id)
            if not factor_def:
                continue

            if category and factor_def.category != category:
                continue

            if computation_type and factor_def.computation_type != computation_type:
                continue

            if query:
                text = f"{factor_def.name} {factor_def.description} {factor_def.factor_id}".lower()
                if query.lower() not in text:
                    continue

            results.append(factor_def.to_dict())

        return results

    def validate_factor(self, factor_id: str, data: pd.DataFrame = None) -> Dict[str, Any]:
        warnings.warn(
            "FactorEngine.validate_factor() is deprecated, use FactorLifecycleService.health_check()",
            DeprecationWarning,
            stacklevel=2,
        )
        result = {
            'factor_id': factor_id,
            'exists': False,
            'valid_definition': False,
            'computable': False,
            'errors': []
        }

        try:
            factor_def = self.storage.load_factor_definition(factor_id)
            if not factor_def:
                result['errors'].append("因子定义不存在")
                return result

            result['exists'] = True
            result['definition'] = factor_def.to_dict()

            required_fields = ['factor_id', 'name', 'computation_type', 'computation_data']
            for field in required_fields:
                if not hasattr(factor_def, field) or getattr(factor_def, field) is None:
                    result['errors'].append(f"缺少必需字段: {field}")

            if not result['errors']:
                result['valid_definition'] = True

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
        warnings.warn(
            "FactorEngine.get_storage_stats() is deprecated, use FactorCatalogService",
            DeprecationWarning,
            stacklevel=2,
        )
        factors = self.storage.list_factors()
        categories = self.list_categories()

        category_stats = {}
        computation_type_stats = {}

        for factor_id in factors:
            factor_def = self.storage.load_factor_definition(factor_id)
            if factor_def:
                cat = factor_def.category
                if cat not in category_stats:
                    category_stats[cat] = 0
                category_stats[cat] += 1

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
        warnings.warn(
            "FactorEngine.export_factor_list() is deprecated, use FactorCatalogService",
            DeprecationWarning,
            stacklevel=2,
        )
        factors_data = []

        for factor_id in self.storage.list_factors():
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


_global_engine = None


def get_global_engine() -> FactorEngine:
    """获取全局引擎实例"""
    global _global_engine
    if _global_engine is None:
        _global_engine = FactorEngine()
    return _global_engine
