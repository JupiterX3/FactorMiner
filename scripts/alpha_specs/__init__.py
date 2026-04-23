"""
qlib158 / qlib360 / WorldQuant 101 因子规格的数据模块。

每个子模块提供 `build_specs() -> List[FactorSpec]`：
- qlib158_specs — 158 个 qlib Alpha158 因子
- qlib360_specs — 30 个精简版 Alpha360 因子（Fibonacci lag）
- wq101_specs   — WorldQuant 101 可实现子集（~40 个）

FactorSpec 字段（dict）：
- factor_id:        唯一 id，小写下划线
- name:             展示名
- description:      文字描述（WQ101 必须包含原始公式与语义折衷说明）
- category:         "qlib158" / "qlib360" / "wq101"
- subcategory:      子分类（如 "kbar" / "rolling" / "price"）
- is_window:        bool，窗口型因子标记
- min_warmup_bars:  int，冷启动需要的 bar 数
- imports:          List[str]，写入函数文件顶部的 import 行
- body:             str，函数定义源码（含 def calculate(...)）
"""

from .qlib158_specs import build_specs as build_qlib158_specs  # noqa: F401
from .qlib360_specs import build_specs as build_qlib360_specs  # noqa: F401
from .wq101_specs import build_specs as build_wq101_specs  # noqa: F401
