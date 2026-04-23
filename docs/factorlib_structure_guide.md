# FactorLib 文件夹结构指南（V4）

## 概述

V4 把因子库一级目录从「技术 vs 挖掘」改成**按数据来源分组**。这样：
- 目录名本身就表明「这类因子需要什么数据」，映射到数据下载器与实时 API；
- 新数据源接入 = 新目录 = 新一级分类，不需要改代码；
- 挖掘因子 / 事件因子等细分语义下沉到二级 `subcategory` 字段。

## 新的目录结构

```
factorlib/
├── basic_kline/                 # 一级：仅需 OHLCV
│   ├── definitions/             #   因子定义（JSON）
│   ├── functions/               #   Python 函数（.py）
│   ├── formulas/                #   公式（.txt，可选）
│   ├── evaluations/             #   评估结果
│   └── mining_history/          #   GP/RL 挖掘会话与结果
│
├── derivatives/                 # 一级：衍生品微观结构（OI/LSR/taker/basis）
│   ├── definitions/
│   ├── functions/
│   ├── formulas/
│   └── evaluations/
│
├── funding/                     # 一级：资金费率
│   ├── definitions/
│   ├── functions/
│   ├── formulas/
│   └── evaluations/
│
├── trained_models/              # WebUI 训练出的组合信号模型
├── exports/                     # 因子导出产物
└── *_archived_YYYYMMDD/         # 历史目录归档，运行时被忽略
```

> 新增一级分类（例如链上数据 `onchain/`）：直接新建同构子目录即可，
> `TransparentFactorStorage.list_source_groups()` 会自动发现。

## 一级分类说明

### basic_kline —— 仅需 OHLCV
- 典型因子：MA/RSI/MACD、Alpha158/Alpha360、WorldQuant Alpha101、GP 挖掘因子。
- 数据依赖：`data/binance/<trade_type>/*.feather`。
- 实时可行性：所有主流交易所 K 线 API 都直接支持。

### derivatives —— 衍生品微观结构
- 典型因子：持仓量变化、多空账户比、大户持仓比、主动买入占比、基差 Z-Score。
- 数据依赖：`data/binance/futures_metrics/`、`futures_markprice/`、`futures_indexprice/`。
- 实时可行性：Binance 提供 `/futures/data/*` 与 premiumIndex 接口（分钟级）。

### funding —— 资金费率
- 典型因子：当前费率、MA、Z-Score、累计、换手。
- 数据依赖：`data/binance/futures_funding/`。
- 实时可行性：Binance `fapi/v1/fundingRate`、`premiumIndex` 实时可取。

## 二级分类 `subcategory`

写在因子 JSON 里，纯标签，不再决定目录。常用值：

| subcategory         | 含义                                            |
| ------------------- | ----------------------------------------------- |
| `technical`         | 经典技术指标                                    |
| `event`             | 事件型 0/1 信号（突破/交叉等，截面评估会过滤） |
| `mined`             | GP / RL / 手动入库的挖掘因子                    |
| `qlib158` / `qlib360` | qlib Alpha 因子族                              |
| `wq101`             | WorldQuant Alpha101                             |
| `derivatives_flow`  | 衍生品资金流/多空结构类                         |
| `funding_level`     | 资金费率水平                                    |

## 因子定义示例

```json
{
  "factor_id": "lsr_top_position_zscore",
  "name": "大户持仓多空比 Z-Score",
  "description": "top trader position LSR 的滚动 Z-Score",
  "category": "derivatives",
  "subcategory": "derivatives_flow",
  "computation_type": "function",
  "computation_data": {
    "function_file": "derivatives/functions/lsr_top_position_zscore.py",
    "entry_point": "calculate"
  },
  "parameters": {"window": 96},
  "output_type": "series"
}
```

- `category` 必须等于一级目录名；保存时引擎会据此写入对应目录。
- `computation_type` 只保留三种：
  - `function`：调用 `functions/<id>.py` 的 `calculate(data, **params)`；
  - `formula`：`formulas/<id>.txt` 中的一行表达式；
  - `ml_model`：**算法代理**，加载 `user_algo/<algorithm>/` 模块计算当前 symbol。
    不再承载静态 pickle 模型（ML 预训练因子已全部移除）。

## 存储系统（TransparentFactorStorage）

关键行为变化：

```python
class TransparentFactorStorage:
    DEFAULT_SOURCE_GROUPS = ('basic_kline', 'derivatives', 'funding')
    _EXCLUDED_NAME_TOKENS = ('_archived_', '_deprecated', '_backup')

    def list_source_groups(self) -> list[str]:
        """动态扫描 factorlib/ 下所有含 definitions/ 的目录；
        默认分组优先，其它按字母序；归档/备份目录自动忽略。"""

    def _group_for_category(self, category: str) -> str:
        """根据 category 决定写入哪一级目录；category 不在默认组时
        直接使用 category 作为目录名，支持无侵入新增一级分类。"""
```

- `save_function_factor()` 是唯一对外的保存入口；老 API
  `save_technical_factor` / `save_minactor_factor` 保留为兼容别名。
- 已移除：`save_ml_model_factor`、`save_ml_factor`、`save_model`、
  `load_model`（全部服务于已删除的 ML 预训练因子）。
- `compute_factor()` 对 `ml_model` 走算法代理路径，不再加载 `.pkl`。

## 数据加载（DataLoader.load_with_extras）

跨一级分类因子必须通过新的加载方式，一次 join 好所有列：

```python
from factor_miner.core.data_loader import DataLoader

df = DataLoader().load_with_extras(
    symbol='BTC_USDT_USDT',
    interval='1h',
    start_date='2024-01-01',
    end_date='2024-12-31',
    include=['metrics', 'funding', 'basis'],
    trade_type='futures',
)
```

`include` 取值：
- `metrics` → 持仓量 / 多空比 / 主动买入占比；
- `funding` → 资金费率；
- `basis` → 基差（自动由 mark/index 计算得到 `basis`）。

## 迁移与归档

`scripts/migrate_factorlib_v4.py` 已经执行过（dry-run + apply）：

- `technicals/` → 按 `subcategory`（`technical` / `event`）迁入 `basic_kline/`；
- `minactors/` 中 GP/RL 挖掘结果 → `basic_kline/` + `subcategory=mined`；
- 29 个 ML 预训练因子（含 JSON/函数/pkl）已删除，清单见
  `factorlib/exports/deleted_factors_manifest.json`；
- 老目录重命名为 `technicals_archived_YYYYMMDD/` / `minactors_archived_YYYYMMDD/`，
  运行时不会被扫描。

## 原则回顾

1. **目录 = 数据来源**：看目录名就知道这类因子需要什么数据、能否实时跑。
2. **源码可读**：每个因子都有 JSON + .py/.txt，不允许二进制黑盒。
3. **可扩展**：加数据源就是加目录，存储/前端都是动态发现。
4. **ML 路径单一**：只保留"算法代理"一种 ML 形态（GP/RL 挖掘），
   组合信号模型统一走 `trained_models/` 路径。

---
*最后更新：2026-04-23*
