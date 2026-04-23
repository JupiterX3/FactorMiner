# 工单：截面评估 K 线因子库扩展 — qlib Alpha158 / Alpha360 / WorldQuant 101

- **创建日期**：2026-04-22
- **作者**：FactorMiner
- **关联审计**：`audit/截面评估代码review.md`
- **目标页面**：`/cross_sectional_evaluation`

---

## 0. 决策前提（默认已锁定）

| # | 决策点 | 选择 | 影响 |
|---|---|---|---|
| D1 | Alpha360 范围 | **精简 30 个**（Fibonacci lag + 比率） | 避免 330 个冗余 close_lag 因子污染因子列表 |
| D2 | WQ101 外层 `rank` 语义 | **允许去掉外层 `rank`**，由截面评估自己排序 | 可落地数量从 ~15 提升到 ~40 |
| D3 | 前端 `renderFactors` | **新增 3 个子分组**（qlib158 / qlib360 / wq101） | 可按族折叠/全选，UI 更清晰 |
| D4 | audit Bug 是否同修 | **同步修复 #1 前瞻偏差**（`ensemble_backtest`） | 新因子评估结果可信 |

---

## 1. 任务总览

| 期次 | 交付物 | 规模 | 预估工时 |
|---|---|---|---|
| 第 1 期 | 脚手架（`_alpha_ops.py` + 注册脚本 + `is_window` 元数据 + 前端兼容）| 约 6 个文件新增 / 小改 | 1–2 天 |
| 第 2 期 | qlib Alpha158 全量 | 158 对 JSON+PY | 2 天 |
| 第 3 期 | qlib Alpha360 精简 | 30 对 JSON+PY | 1 天 |
| 第 4 期 | WorldQuant 101 可行子集 | 约 40 对 JSON+PY | 2–3 天 |
| 伴随 | 前端 3 个子分组 + ensemble_backtest bug 修复 | 2 个文件小改 | 0.5 天 |

---

## 2. 目录与文件产出约定

```
factorlib/technicals/
├── functions/
│   ├── _alpha_ops.py                  # 共享算子（新增）
│   ├── alpha158_<name>.py             # 158 个（新增）
│   ├── alpha360_<field>_lag_<n>.py    # 30 个（新增）
│   └── wq101_alpha<NNN>.py            # ~40 个（新增）
├── definitions/
│   └── 与上述 .py 同名的 .json 文件
└── （现有文件不动）

scripts/
└── register_qlib_wq_factors.py        # 一键注册/重注册脚本（新增）

docs/
├── qlib158_factor_list.md             # 158 个因子清单与公式（新增）
├── qlib360_disabled_full.md           # 未落地的 330 个 lag 说明（新增）
└── wq101_compat_notes.md              # WQ101 语义折衷与未实现清单（新增）
```

**命名规则**：
- qlib158：`alpha158_kmid`、`alpha158_ma_5`、`alpha158_beta_20` …
- qlib360：`alpha360_close_lag_5`、`alpha360_vwap_lag_21` …
- WQ101：`wq101_alpha001`、`wq101_alpha002` …

**`factor_id` 规则**：全小写、下划线分隔，与文件名完全一致。

---

## 3. 第 1 期：脚手架

### 3.1 `factorlib/technicals/functions/_alpha_ops.py`

提供统一算子，所有新 alpha 从此处导入：

| 算子 | 签名 | qlib/WQ 对应 | 备注 |
|---|---|---|---|
| `ts_mean(s, n)` | 滚动均值 | `Mean` / `sum(x,n)/n` | |
| `ts_std(s, n)` | 滚动标准差 | `Std` / `stddev` | |
| `ts_sum(s, n)` | 滚动求和 | `Sum` / `sum` | |
| `ts_max(s, n)` | 滚动最大 | `Max` / `ts_max` | |
| `ts_min(s, n)` | 滚动最小 | `Min` / `ts_min` | |
| `ts_rank(s, n)` | 滚动 pct rank | `Rank` / `ts_rank` | 最后一点在窗口内的分位 |
| `ts_argmax(s, n)` | 滚动 argmax（位置）| `Idxmax` / `ts_argmax` | 1-based |
| `ts_argmin(s, n)` | 滚动 argmin（位置）| `Idxmin` / `ts_argmin` | 1-based |
| `ts_corr(a,b,n)` | 滚动相关 | `Corr` / `correlation` | |
| `ts_cov(a,b,n)` | 滚动协方差 | `Cov` / `covariance` | |
| `delta(s, n)` | 差分 | `delta` | `s - s.shift(n)` |
| `ref / delay(s, n)` | 滞后 | `Ref` / `delay` | `s.shift(n)` |
| `decay_linear(s, n)` | 线性加权衰减 | `WMA` / `decay_linear` | 权重 1..n 归一 |
| `signed_power(s, p)` | 带符号幂 | `SignedPower` | |
| `sign(s)` | 符号 | `sign` | |
| `log(s)` | 自然对数 | `Log` | 0 → NaN |
| `scale(s, k=1)` | L1 归一 | `scale` | |
| `adv(vol, n=20)` | 平均成交量 | `adv20` | `ts_mean(vol, n)` |
| `vwap_proxy(data)` | VWAP 代理 | `vwap` | `(H+L+C)/3`（无逐笔数据近似）|
| `returns(close)` | 收益率 | `returns` | `close.pct_change()` |
| `rolling_beta(y, x, n)` | 回归系数 | qlib `BETA` | 对时间回归即 x=range |
| `rolling_rsqr(y, n)` | 拟合优度 | qlib `RSQR` | |
| `rolling_resi(y, n)` | 回归残差当前值 | qlib `RESI` | |
| `ts_rank_pct(s, n)` | 同 `ts_rank` 别名 | | 便于阅读 |

> **注意**：`_alpha_ops.py` 名字以下划线开头，不与任何 `factor_id` 冲突。`FactorEngine._compute_function_factor` 只会对 `factor_id.py` 做动态加载，不会把 `_alpha_ops.py` 当成因子。

### 3.2 `scripts/register_qlib_wq_factors.py`

一次性注册脚本，功能：
1. 读取内部字典 `ALPHA158_SPECS` / `ALPHA360_SPECS` / `WQ101_SPECS`（每条含 `id/name/desc/subcategory/is_window/min_warmup/code`）
2. 对每条生成：
   - `factorlib/technicals/functions/<id>.py`（函数体直接从 spec 的 `code` 字段取，顶部 `from _alpha_ops import *`）
   - `factorlib/technicals/definitions/<id>.json`
3. CLI：
   ```
   python scripts/register_qlib_wq_factors.py --family qlib158     # 只注册 158
   python scripts/register_qlib_wq_factors.py --family qlib360     # 只注册 360 精简
   python scripts/register_qlib_wq_factors.py --family wq101
   python scripts/register_qlib_wq_factors.py --family all
   python scripts/register_qlib_wq_factors.py --family all --overwrite    # 重写已存在的
   python scripts/register_qlib_wq_factors.py --dry-run                    # 只打印不写盘
   ```
4. 退出前打印 `总因子数 / 成功 / 跳过（已存在）/ 失败`

### 3.3 `is_window` 元数据贯通

- `factor_miner/core/factor_storage.py::FactorDefinition` — 不改（`metadata` 本身就是 `Dict`）
- `webui/routes/factors.py::list_factors` — **小改**：在 `processed_factors.append(...)` 里加：
  ```python
  'is_window': data.get('metadata', {}).get('is_window'),
  'min_warmup_bars': data.get('metadata', {}).get('min_warmup_bars'),
  ```
- `webui/templates/cross_sectional_evaluation.html::isWindowBasedKlineFactor` — **小改**：优先读 `factor.is_window`，为 `null/undefined` 时才降级到原启发式（保持对 300+ 老因子兼容）。

### 3.4 接受度测试（烟雾）

```bash
python scripts/register_qlib_wq_factors.py --family qlib158 --dry-run
```
输出行数 ≥ 158，无异常。

```python
from factor_miner.core.factor_engine import get_global_engine
eng = get_global_engine()
eng.compute_single_factor('alpha158_kmid', test_df)  # 返回 pd.Series
```

---

## 4. 第 2 期：qlib Alpha158

**参考**：`qlib/contrib/data/loader.py::Alpha158DL.get_feature_config` 的 `KBAR / PRICE / VOLUME / ROLLING` 四大块。

### 4.1 K-Bar（9 个，`subcategory="kbar"`, `is_window=false`）

| factor_id | 公式 |
|---|---|
| `alpha158_kmid`  | `(close-open)/open` |
| `alpha158_klen`  | `(high-low)/open` |
| `alpha158_kmid2` | `(close-open)/(high-low+1e-12)` |
| `alpha158_kup`   | `(high-max(open,close))/open` |
| `alpha158_kup2`  | `(high-max(open,close))/(high-low+1e-12)` |
| `alpha158_klow`  | `(min(open,close)-low)/open` |
| `alpha158_klow2` | `(min(open,close)-low)/(high-low+1e-12)` |
| `alpha158_ksft`  | `(2*close-high-low)/open` |
| `alpha158_ksft2` | `(2*close-high-low)/(high-low+1e-12)` |

### 4.2 Price（5 lag × 4 字段 = 20 个，`subcategory="price"`, `is_window=false`）

字段 × lag 组合：`{open, high, low, vwap} × {0, 1, 2, 3, 4}` → `<field>_lag_<n> / close` 形式。

> 注：qlib 原版 lag 从 0 开始，但 lag=0 时 open/high/low/vwap 相对 close 是当根信息，**需 `shift(trade_shift=1)` 由评估层处理**，因子内部不 shift。

### 4.3 Volume（5 个，`subcategory="volume"`, `is_window=false`）

`alpha158_vol_lag_0..4 = volume.shift(n) / (volume + 1e-12)` — 其中 lag_0 恒为 1，qlib 原版保留是为了形状对齐；我们**只生成 lag_1..4**（4 个）+ `alpha158_vwma_5`（1 个，`sum(volume*close,5)/sum(volume,5)`），合计 5 个。

### 4.4 Rolling（124 个，`subcategory="rolling"`, `is_window=true`）

窗口 `w ∈ {5, 10, 20, 30, 60}`（5 档），每档包含：

| 名称 | 公式 | 个数 |
|---|---|---|
| `ma_w`    | `ts_mean(close, w) / close` | 5 |
| `std_w`   | `ts_std(close, w) / close` | 5 |
| `beta_w`  | `(close - close.shift(w)) / (w * close)` | 5 |
| `rsqr_w`  | 对 `close` 关于时间的滚动回归 R² | 5 |
| `resi_w`  | `rolling_resi(close, w) / close` | 5 |
| `max_w`   | `ts_max(high, w) / close` | 5 |
| `min_w`   | `ts_min(low, w) / close` | 5 |
| `qtlu_w`  | `high.rolling(w).quantile(0.8) / close` | 5 |
| `qtld_w`  | `low.rolling(w).quantile(0.2) / close` | 5 |
| `rank_w`  | `ts_rank(close, w)` | 5 |
| `rsv_w`   | `(close - ts_min(low, w)) / (ts_max(high, w) - ts_min(low, w) + 1e-12)` | 5 |
| `imax_w`  | `ts_argmax(high, w) / w` | 5 |
| `imin_w`  | `ts_argmin(low, w) / w` | 5 |
| `imxd_w`  | `(ts_argmax(high, w) - ts_argmin(low, w)) / w` | 5 |
| `corr_w`  | `ts_corr(close, log(volume+1), w)` | 5 |
| `cord_w`  | `ts_corr(close/close.shift(1), log(volume/volume.shift(1)+1), w)` | 5 |
| `cntp_w`  | `((close > close.shift(1)).rolling(w).mean())` | 5 |
| `cntn_w`  | `((close < close.shift(1)).rolling(w).mean())` | 5 |
| `cntd_w`  | `cntp_w - cntn_w` | 5 |
| `sump_w`  | `ts_sum(max(close-close.shift(1),0), w) / (ts_sum(abs(close-close.shift(1)), w)+1e-12)` | 5 |
| `sumn_w`  | `1 - sump_w` | 5 |
| `sumd_w`  | `2*sump_w - 1` | 5 |
| `vma_w`   | `ts_mean(volume, w) / (volume+1e-12)` | 5 |
| `vstd_w`  | `ts_std(volume, w) / (volume+1e-12)` | 5 |
| `wvma_w`  | `ts_std(abs(close/close.shift(1)-1)*volume, w) / (ts_mean(abs(close/close.shift(1)-1)*volume, w)+1e-12)` | 5 |

共 25 类 × 5 窗口 = **125**；去掉与 qlib 重复定义的 1 个，实际 **124**。加上 §4.1/§4.2/§4.3 共 9 + 20 + 5 + 124 = **158**。

> 实际实现时按 qlib 源码精确对齐；本表作为自检清单。

### 4.5 `min_warmup_bars` 标注

- KBAR: 1
- Price/Volume lag: `max(lag) + 1 = 5`
- Rolling: `max(w) = 60`

评估层后续可按此跳过 warmup 截面（本期不接，只标元数据）。

---

## 5. 第 3 期：qlib Alpha360 精简

### 5.1 Fibonacci lag（27 个）

| 字段 | lag 集合 | 个数 |
|---|---|---|
| close | {1, 2, 3, 5, 8, 13, 21, 34, 55} | 9 |
| open  | {1, 5, 21} | 3 |
| high  | {1, 5, 21} | 3 |
| low   | {1, 5, 21} | 3 |
| vwap  | {1, 5, 21} | 3 |
| volume| {1, 5, 21} | 3 |
| 比率 `close_lag_1/close_lag_20`、`volume_lag_1/volume_lag_20` 等 | — | 3 |
| **合计** | | **27** |

公式：`alpha360_<field>_lag_<n> = field.shift(n) / close`（volume 分母用 `ts_mean(volume, 1)` 即 volume 自身+ε）。

### 5.2 补齐到 30 的 3 个辅助因子

| factor_id | 公式 | 说明 |
|---|---|---|
| `alpha360_close_ratio_1_20` | `close.shift(1) / close.shift(20)` | 中期动量 |
| `alpha360_volume_ratio_1_20` | `volume.shift(1) / volume.shift(20)` | 中期量能 |
| `alpha360_hl_range_lag_5` | `(high.shift(5) - low.shift(5)) / close` | 5 日前振幅 |

### 5.3 留档

在 `docs/qlib360_disabled_full.md` 写明：
- 未落地的 333 个为什么砍（冗余度 ρ > 0.99）
- 如何通过 `register_qlib_wq_factors.py --family qlib360 --full` 一键生成（预留 `--full` 参数）

---

## 6. 第 4 期：WorldQuant 101 可行子集

### 6.1 落地清单（~40 个）

**严格遵循原公式但去掉最外层 `rank`（若有）的 alpha**（标 🟢）：

`alpha001, 002, 003, 004, 006, 007, 008, 009, 010, 012, 013, 014, 015, 019, 020, 021, 022, 024, 028, 029, 032, 034, 035, 037, 038, 040, 043, 046, 049, 050, 051, 053, 054, 055, 065, 068, 071, 072, 078, 083, 084, 085, 094, 099, 101`

以上 45 个中，在单币场景下经逐条核对，**可落地**约 40 个（排除掉 028/029 这类依赖 `adv20` 且需要 `indneutralize` 的硬阻塞项，具体名单在第 4 期开工时逐条敲定并在 `docs/wq101_compat_notes.md` 中记录）。

### 6.2 语义折衷规则

| 原表达 | 替换为 | 备注 |
|---|---|---|
| 最外层 `rank(x)` | 直接返回 `x` | 截面评估自会排名 |
| 中间层 `rank(x)` | `ts_rank(x, 10)` | 时序 pct rank 近似，描述里标注 |
| `indneutralize(x, g)` | 直接返回 `x` | 加密市场无行业映射 |
| `adv20` | `ts_mean(volume, 20)` | 单币 |
| `vwap` | `vwap_proxy(data)` | `(H+L+C)/3` |
| `returns` | `close.pct_change()` | |
| `cap` / `IndClass.*` | **直接放弃整个 alpha** | |

### 6.3 JSON `description` 必填项

每个 WQ101 因子的 JSON 描述中必须包含：
1. 原始公式（作为字符串）
2. 做过的语义折衷（如"已去掉最外层 rank"、"rank → ts_rank(x,10)"）
3. 与原始 WQ101 的行为差异程度（`high/medium/low`）

### 6.4 未落地清单

剩余 ~55 个 alpha 在 `docs/wq101_compat_notes.md` 单独列表：
- 依赖 `cap` 的
- 依赖 `IndClass.*` 的
- 嵌套超过 2 层 `rank` 且拆解后语义偏离过大的

---

## 7. 伴随任务

### 7.1 前端 `renderFactors` 新增 3 个子分组

文件：`webui/templates/cross_sectional_evaluation.html`

在 `categories['basic_kline'].subgroups` 下新增：

```javascript
basic_kline_qlib158: { name: 'qlib158 因子', factors: [] },
basic_kline_qlib360: { name: 'qlib360 因子（精简）', factors: [] },
basic_kline_wq101:   { name: 'WorldQuant 101（可实现子集）', factors: [] },
```

分流逻辑（在 `factors.forEach(f => ...)` 里）：

```javascript
const fid = String(f.id || '').toLowerCase();
if (fid.startsWith('alpha158_')) {
    categories['basic_kline'].factors.push(f);
    categories['basic_kline'].subgroups.basic_kline_qlib158.factors.push(f);
    return;
}
if (fid.startsWith('alpha360_')) {
    categories['basic_kline'].factors.push(f);
    categories['basic_kline'].subgroups.basic_kline_qlib360.factors.push(f);
    return;
}
if (fid.startsWith('wq101_')) {
    categories['basic_kline'].factors.push(f);
    categories['basic_kline'].subgroups.basic_kline_wq101.factors.push(f);
    return;
}
// 原有 single_bar / window 逻辑兜底
```

同时给 `basic_kline` 的渲染模板加上这 3 个子组的"全选"toggle 与因子列表渲染（参照现有 `basic_kline_single_bar` / `basic_kline_window` 的模板复制）。

### 7.2 修复 audit #1 — `ensemble_backtest` 前瞻偏差

文件：`webui/routes/factors.py:1033`（按 audit 报告定位）

```diff
- returns = market_data['close'].pct_change().shift(-1)
+ returns = market_data['close'].pct_change()
```

> 与截面评估的口径对齐（`prepare_cross_sectional_data` 里 `future_returns` 已经做过 `.shift(-predict_step)`，权重端不应再用未来收益）。

此项在第 1 期同步修复，避免新因子评估入库后用到错误的组合回测。

---

## 8. 质量门槛（每期必过）

| 检查项 | 工具 | 标准 |
|---|---|---|
| 新增 `.py` 无 import 错误 | `python -c "import importlib; importlib.import_module('factorlib.technicals.functions.alpha158_kmid')"` | 0 错误 |
| 数据尾部 NaN 检测 | 脚本：尾部注入 NaN 后因子输出尾部必须 NaN | 0 未来数据泄露 |
| 单币种冒烟 | `FactorEngine.compute_single_factor` | 每因子至少产出 非空 + `finite` 比例 ≥ 80% |
| 截面烟雾（5 币 × 1 月 × 1h） | `cross_sectional_evaluate` 路由 | 每因子 `n_periods_ic >= 50` |
| `list_factors` 冷启动 | 手动刷新接口 | 响应 ≤ 3 秒（当前约 ~260 因子增量，基线 ~2 秒）|

---

## 9. 回滚策略

- 注册脚本 `--overwrite` 默认 `False`；一期内如发现错误，直接删除对应 `functions/<id>.py` 和 `definitions/<id>.json` 即可，**不会影响**其它因子。
- 前端改动为小范围新增 `if` 分支，对老因子的渲染路径无影响。
- `ensemble_backtest` 修复是单行 diff，可独立回滚。

---

## 10. 进度追踪

使用 TodoWrite 并行跟踪，7 个任务：

1. ✅ 生成工单
2. ⏳ 第 1 期：脚手架
3. ⏳ 第 2 期：qlib Alpha158
4. ⏳ 第 3 期：qlib Alpha360 精简
5. ⏳ 第 4 期：WorldQuant 101
6. ⏳ 前端 3 个子分组
7. ⏳ `ensemble_backtest` bug 修复

每期完成后更新本工单末尾的"变更记录"。

---

## 变更记录

| 日期 | 作者 | 内容 |
|---|---|---|
| 2026-04-22 | FactorMiner | 初稿 |
