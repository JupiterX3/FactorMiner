"""
核心模块
包含因子挖掘的核心功能
"""

from .data_loader import DataLoader
from .factor_builder import FactorBuilder
from .factor_catalog import FactorCatalogService
from .factor_evaluator import FactorEvaluator
from .factor_executor import FactorExecutor
from .factor_lifecycle import FactorLifecycleService
from .factor_optimizer import FactorOptimizer
from .factor_repository import FactorRepository
from .factor_schema import CleanupResult, EvaluationAggregation, FactorDefinition, FactorQuery, FactorSummary, HealthIssue, HealthReport
from .gp_miner import GPMiner, run_cross_sectional_mining

TORCH_AVAILABLE = False
try:
    from .rl_miner import RLMiner, run_rl_cross_sectional_mining
    TORCH_AVAILABLE = True
except ImportError:
    RLMiner = None
    run_rl_cross_sectional_mining = None

__all__ = [
    'DataLoader',
    'CleanupResult',
    'EvaluationAggregation',
    'FactorBuilder',
    'FactorCatalogService',
    'FactorDefinition',
    'FactorEvaluator',
    'FactorExecutor',
    'FactorLifecycleService',
    'FactorOptimizer',
    'FactorQuery',
    'FactorRepository',
    'FactorSummary',
    'HealthIssue',
    'HealthReport',
    'GPMiner',
    'run_cross_sectional_mining',
    'RLMiner',
    'run_rl_cross_sectional_mining',
    'TORCH_AVAILABLE',
] 
