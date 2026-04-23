# FactorMiner 因子库（V4 架构）

## 设计目标
- 因子 = 算法（不落地因子数值，计算即数据）
- 逻辑完全透明、可读、可审计（JSON 定义 + 源码函数/公式）
- **一级目录 = 数据来源**，方便直接看出因子对数据的要求与实时可行性
- 统一引擎、统一存储、统一调用方式

## 目录结构（V4）
```
factorlib/
├── basic_kline/          # 一级分类 1：仅需 OHLCV 即可计算
│   ├── definitions/      #   因子定义（JSON）
│   ├── functions/        #   Python 因子函数（.py）
│   ├── formulas/         #   公式（.txt，可选）
│   ├── evaluations/      #   评估结果（每因子一份）
│   └── mining_history/   #   挖掘（GP/RL）产出的会话与结果
│
├── derivatives/          # 一级分类 2：需要衍生品微观结构数据
│   ├── definitions/      #   持仓量 / 多空比 / 主动买入 / 基差等
│   ├── functions/
│   ├── formulas/
│   └── evaluations/
│
├── funding/              # 一级分类 3：需要资金费率历史
│   ├── definitions/
│   ├── functions/
│   ├── formulas/
│   └── evaluations/
│
├── trained_models/       # WebUI 训练的组合信号模型（LightGBM 等）
├── exports/              # 因子导出产物
└── (其它 *_archived_*)    # 历史归档，运行时会被忽略
```

> 加新一级分类（比如将来接入链上数据 `onchain/`）：直接新建同构目录 +
> 子目录 `definitions/functions/evaluations/` 即可，`TransparentFactorStorage`
> 会自动发现；前端分类也会基于 `data_requirement` 动态渲染。

## 二级分类（subcategory，写在定义 JSON 里）
一级分类固定是数据来源；每个因子可以再标 `subcategory` 做细分，比如：
- `technical`：经典技术指标（MA/RSI/MACD …）
- `event`：事件型 0/1 信号（突破/交叉等），在截面评估里会被筛掉
- `mined`：GP / RL / 手动入库的挖掘因子
- `qlib158` / `qlib360` / `wq101`：来自 qlib 或 WorldQuant Alpha 的因子族

## 因子定义（`<group>/definitions/*.json`）
```json
{
  "factor_id": "oi_change_pct",
  "name": "持仓量变化率",
  "category": "derivatives",
  "subcategory": "derivatives_flow",
  "computation_type": "function",
  "computation_data": {
    "function_file": "derivatives/functions/oi_change_pct.py",
    "entry_point": "calculate"
  },
  "parameters": {"window": 5},
  "output_type": "series"
}
```
- `category`：数据来源 = 一级目录名；必须和文件所在目录一致。
- `subcategory`：细分标签（可选）。
- `computation_type`：`function` / `formula` / `ml_model`（算法代理，目前仅 GP/RL 挖掘链路使用）。

## 计算方式

### 1. 公式因子（formula）
```text
# factorlib/basic_kline/formulas/sma.txt
close.rolling(window=period).mean()
```

### 2. 函数因子（function）
```python
# factorlib/derivatives/functions/basis_zscore.py
def calculate(data, window=60, **kwargs):
    basis = data['basis']
    mean = basis.rolling(window).mean()
    std = basis.rolling(window).std()
    return (basis - mean) / std
```
`data` 是包含所需列的 `pd.DataFrame`。对于 `derivatives` / `funding` 类因子，
必须通过 `DataLoader.load_with_extras(include=[...])` 加载，以便额外列被 join 进来。

### 3. 算法代理（ml_model）
仅用于 GP / RL 挖掘出来的因子：JSON 指向 `user_algo/` 下的某个算法模块，
引擎在运行时调用该模块的 `calculate_single_factor` 计算当前 symbol。
**不再使用静态 pickle 模型**（已全部清理）。

## 数据加载（新增 extra-data 支持）
```python
from factor_miner.core.data_loader import DataLoader

loader = DataLoader()
df = loader.load_with_extras(
    symbol='BTC_USDT_USDT',
    interval='1h',
    start_date='2024-01-01',
    end_date='2024-12-31',
    include=['metrics', 'funding', 'basis'],   # 按需 join
    trade_type='futures',
)
# df 列：open/high/low/close/volume/taker_buy_base/taker_buy_quote +
#        open_interest/lsr_global/lsr_top_account/lsr_top_position/... +
#        funding_rate + mark_close/index_close/basis
```

数据落盘结构：
```
data/binance/
├── futures/                 # 原始 OHLCV（含 taker_buy_*）
├── futures_metrics/         # OI / LSR / taker_buy_ratio
├── futures_funding/         # 资金费率历史
├── futures_markprice/       # 标记价 K 线
└── futures_indexprice/      # 指数价 K 线
```

## 调用方式
```python
from factor_miner.core.factor_engine import get_global_engine
from factor_miner.core.factor_storage import get_global_storage

engine = get_global_engine()
storage = get_global_storage()

# 列出所有一级分类（动态扫描，自动忽略 *_archived_*）
print(storage.list_source_groups())        # ['basic_kline', 'derivatives', 'funding', ...]

# 计算一个因子
series = engine.compute_single_factor('basis_zscore', df)
```

## 原则
- 零「注册器」、零「热插拔缓存」：一切定义来自 JSON，引擎启动即可见。
- 因子逻辑必须以源码/公式文本形式入库，禁止二进制黑盒。
- 一级目录 ≡ 数据来源；加分类就是加目录，不改代码。
- `ml_model` 仅作为「算法代理」使用，不再承载静态预训练因子。

---
*最后更新：2026-04-23*

