# 截面评估功能 代码审计报告

## 审计范围

| 文件 | 角色 | 行数 |
|------|------|------|
| [cross_sectional_evaluation.html](file:///d:/PythonProject/FactorMiner/webui/templates/cross_sectional_evaluation.html) | 前端页面 | 1588 |
| [factors.py](file:///d:/PythonProject/FactorMiner/webui/routes/factors.py) | 后端路由 | 1273 |
| [factor_evaluator.py](file:///d:/PythonProject/FactorMiner/factor_miner/core/factor_evaluator.py) | 核心评估逻辑 | 1209 |
| [factor_engine.py](file:///d:/PythonProject/FactorMiner/factor_miner/core/factor_engine.py) | 因子计算引擎 | 599 |

---

## 🔴 严重问题（Bug / 逻辑错误）

### 1. `ensemble_backtest` 中存在前瞻偏差（Look-Ahead Bias）

[factors.py:L1033](file:///d:/PythonProject/FactorMiner/webui/routes/factors.py#L1033)

```python
returns = market_data['close'].pct_change().shift(-1)  # ← 使用了未来收益！
```

`shift(-1)` 将未来的收益移到了当前时间点。这意味着在创建组合权重（IC加权等）时，**直接使用了未来数据**来决定权重。正确做法应与截面评估保持一致：

```diff
- returns = market_data['close'].pct_change().shift(-1)
+ returns = market_data['close'].pct_change()
```

> [!CAUTION]
> 这是一个典型的前瞻偏差，会导致回测结果严重虚高，无法反映真实的组合性能。

---

### 2. 前端进度计算使用错误的 `total` 基数（续评时）

[cross_sectional_evaluation.html:L823](file:///d:/PythonProject/FactorMiner/webui/templates/cross_sectional_evaluation.html#L823)

```javascript
const totalTasks = isContinue ? (lastRequestData.factor_ids.length) : factorIds.length;
```

当**续评**时，`totalTasks` 取的是原始请求的因子总数。但续评请求 `retryPayload` （L886）已被过滤成只包含剩余未完成的因子ID。后端返回的 `completed` 计数从 1 开始递增（因为后端只看到了这批剩余因子），但前端用**总因子数**做分母，导致：
- 进度条百分比不正确（比如原始 10 个因子完成了 7 个，续评剩余 3 个，后端返回 completed=1/3，但前端显示 `1/10 = 10%`）
- 最终 `done` 事件中 `data.total` 也只是剩余因子数，和 `totalTasks` 不匹配

> [!WARNING]
> 续评场景下进度显示会出现跳跃或不准确。建议在 `handleSSEEvent` 中使用 `completedFactorIds.size` 作为累计已完成数。

---

### 3. `factorIcDict` 反向因子逻辑错误

[cross_sectional_evaluation.html:L1474-L1477](file:///d:/PythonProject/FactorMiner/webui/templates/cross_sectional_evaluation.html#L1474-L1477)

```javascript
const factorIcDict = {};
selectedFactors.forEach(f => {
    factorIcDict[f.id] = autoReverse ? f.ic : Math.abs(f.ic);
});
```

当 `autoReverse = true` 时，传的是原始 IC 值（可能为负），后端 (L1011) 用 `if ic_value < 0` 来判断是否取反。但当 `autoReverse = false` 时，前端传的是 `Math.abs(f.ic)`，**永远为正**，这意味着后端即使收到本应反转的因子也不会反转。

这虽在 `autoReverse=true`（默认）时无问题，但 `autoReverse=false` 的语义应该是"不反转任何因子"，而不是"所有因子都取绝对值"。当前实现正好符合了这个意图，但 `factorIcDict` 中存储的值**失去了原始IC信息**，导致后端 `reversed_factors` 列表永远为空。

> [!IMPORTANT]
> 建议统一语义：`autoReverse=false` 时直接传原始 IC 值并在后端跳过反转逻辑，而非用 `Math.abs` 篡改 IC 值。

---

## 🟡 潜在问题（鲁棒性 / 性能）

### 4. `ThreadPoolExecutor` 中 CPython GIL 限制

[factors.py:L610](file:///d:/PythonProject/FactorMiner/webui/routes/factors.py#L610)

```python
max_workers = min(len(factor_ids), os.cpu_count() or 4, 8)
```

因子计算（pandas/numpy 运算）主要是 **CPU 密集型**。Python `ThreadPoolExecutor` 在 CPython 下受 GIL 限制，**多线程并不能真正并行执行 CPU 密集型纯 Python 代码**。

但要注意：
- pandas 和 numpy 的很多底层操作（如 `np.corrcoef`、`pd.qcut`）实际上在 C 层释放了 GIL
- 线程池仍有一个好处：当一个线程在做 IO（文件读取）时，其他线程可以做计算

> [!NOTE]
> 当前方案在实际场景中可能仍有一定的并行效果（IO + C 扩展释放 GIL），但若想获得真正的 CPU 并行，应考虑 `ProcessPoolExecutor` 或 `multiprocessing`。需权衡进程间数据序列化的开销。

---

### 5. `data_dict` 在多线程间共享但 `market_data.copy()` 不完整

[factor_evaluator.py:L872](file:///d:/PythonProject/FactorMiner/factor_miner/core/factor_evaluator.py#L872)

```python
market_data = market_data.copy().sort_index()
market_data['returns'] = market_data['close'].pct_change()
```

`prepare_cross_sectional_data` 中对 `market_data` 做了 `.copy()` 后再修改（添加 `returns` 列），这是正确的。但在后端路由 `cross_sectional_evaluate` 中，多个线程共享同一个 `data_dict`，而 `evaluate_one_factor → cs_evaluator.evaluate_cross_sectional → prepare_cross_sectional_data` 的链路中确实调用了 `.copy()`，因此**当前不存在竞态条件**。

✅ 线程安全已正确处理。

---

### 6. `FactorEngine.compute_single_factor` 中 `importlib` 动态加载的线程安全问题

[factor_engine.py:L73-L113](file:///d:/PythonProject/FactorMiner/factor_miner/core/factor_engine.py#L73-L113)

```python
spec = importlib.util.spec_from_file_location(...)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

每次调用 `compute_single_factor` 都会重新加载模块文件。在多线程环境下：
1. **性能问题**：重复加载同一个因子的 `.py` 文件（磁盘 IO + 解析），对大量因子评估时有明显开销
2. **潜在竞态**：`sys.path.insert(0, ...)` 修改全局状态，多线程并发调用可能导致 `sys.path` 混乱

> [!WARNING]
> 建议：
> - 对已加载的模块做缓存（`dict[factor_id] → module`），避免重复加载
> - 使用 `threading.Lock` 保护 `sys.path` 修改

---

### 7. SSE 流式响应中的心跳间隔可能导致连接被代理关闭

[factors.py:L624-L641](file:///d:/PythonProject/FactorMiner/webui/routes/factors.py#L624-L641)

```python
while pending:
    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
    if not done:
        now = time.time()
        if now - heartbeat_ts >= 1.5:
            yield _sse_event('progress', {...})
            heartbeat_ts = now
        continue
```

当单个因子计算耗时较长（如 1 分钟以上），心跳事件可能仍不足以阻止某些反向代理（如 nginx）的读超时。当前设置 `X-Accel-Buffering: no` 已考虑了 nginx 缓冲，但 `proxy_read_timeout` 默认 60s 仍可能触发。

✅ 心跳实现基本正确，但在生产环境中需同步配置代理超时参数。

---

### 8. CSV 导出中 `formatNumber`/`formatPercent` 导致数据被格式化为字符串

[cross_sectional_evaluation.html:L1183-L1192](file:///d:/PythonProject/FactorMiner/webui/templates/cross_sectional_evaluation.html#L1183-L1192)

```javascript
csv += `${formatNumber(summary.ic_mean ?? ic.ic_mean)},`;
csv += `${formatPercent(summary.long_short_return ?? returns.long_short_return)},`;
```

`formatNumber` 返回带固定小数位的字符串或 `'-'`，`formatPercent` 返回 `xx.xxxx%` 格式的字符串。这会导致：
1. CSV 中 IC 列可能出现 `'-'` 而非空值，Excel 会将其识别为文本
2. 百分比列带 `%` 符号，无法直接做数值运算
3. 再次导入（CSV 导入功能）时，`parseFloat` 对带 `%` 的字符串可以工作（L1275-L1276 中有 `.replace('%', '')`），但对 `'-'` 值返回 `NaN`

> [!TIP]
> 建议在 CSV 导出时使用原始数值而非格式化字符串：
> ```javascript
> csv += `${summary.ic_mean ?? ic.ic_mean ?? ''},`;
> ```

---

## 🟢 代码质量与优化建议

### 9. `_sanitize_for_json` 重复导入 `math`

[factors.py:L714](file:///d:/PythonProject/FactorMiner/webui/routes/factors.py#L714)

```python
def _sanitize_for_json(obj):
    import math  # ← 每次调用都导入
```

`math` 模块已在文件顶部 `import math`（L11），此处重复导入无害但是冗余代码。

---

### 10. `selectAllFactors` 全选和搜索过滤的交互问题

[cross_sectional_evaluation.html:L570-L573](file:///d:/PythonProject/FactorMiner/webui/templates/cross_sectional_evaluation.html#L570-L573)

```javascript
function selectAllFactors() {
    $('.factor-select-checkbox').prop('checked', true);
    updateFactorCount();
}
```

当用户使用搜索框过滤了因子后，点击"全选"会选中**所有因子**（包括被 `display:none` 隐藏的），而不是仅可见的因子。这可能导致用户误选大量不相关的因子。

> [!TIP]
> 修改为仅选可见因子：
> ```javascript
> $('.factor-select-checkbox:visible').prop('checked', true);
> ```

---

### 11. `calculate_cross_sectional_ic` 的截面最小样本阈值为 5 可能过低

[factor_evaluator.py:L930、L940](file:///d:/PythonProject/FactorMiner/factor_miner/core/factor_evaluator.py#L930)

```python
if len(group) < 5:
    continue
```

截面评估中，每个时间点至少需要 5 个币种才计算 IC。对于 Spearman/Pearson 相关系数，5 个样本的统计功效非常低，结果噪声大。建议将阈值提高到 10 或 15。

---

### 12. 年化夏普比率中 `periods_per_year` 默认值偏离实际

[factor_evaluator.py:L1096-L1113](file:///d:/PythonProject/FactorMiner/factor_miner/core/factor_evaluator.py#L1096-L1113)

```python
periods_per_year = 365  # 默认按天
if tf.endswith('h'):
    periods_per_year = (24 / hours) * 365
```

加密货币市场全年 365 天 24 小时交易，使用 365 天计算对于 1h 时间框架得到 8760。此处**逻辑正确**，但如果扩展到传统金融市场需要注意。

✅ 当前场景无问题。

---

### 13. `noranklist` 滑块 `initRangeSlider` 与 `updateRangeSlider` 的重复创建

[cross_sectional_evaluation.html:L736-L762, L764-L792](file:///d:/PythonProject/FactorMiner/webui/templates/cross_sectional_evaluation.html#L736-L792)

`initRangeSlider()` 在页面加载时创建一个默认初始范围（最近1年），但 `loadLocalSymbols` 的回调会立即调用 `updateRangeSlider()` 来销毁并重建。这导致 `initRangeSlider()` 创建的滑块被秒毁，存在短暂的闪烁。

> [!TIP]
> 可延迟初始化或直接在 `loadLocalSymbols` 成功后再创建滑块，避免无谓的创建-销毁循环。

---

### 14. `list_factors` API 性能瓶颈

[factors.py:L43-L153](file:///d:/PythonProject/FactorMiner/webui/routes/factors.py#L43-L153)

每次调用 `list_factors` 都会：
1. 遍历两个目录的所有 `.json` 文件
2. 对每个因子执行 `core_load_evaluations` 读取评估历史，涉及额外文件 IO
3. 聚合计算平均指标

当因子数量较多时（几十到上百），这个 API 的响应时间会线性增长。建议：
- 添加内存缓存（带 TTL）
- 或在评估完成时预计算聚合指标

---

## 📝 总结

| 级别 | 数量 | 说明 |
|------|------|------|
| 🔴 严重 | 3 | 前瞻偏差、进度计算错误、反向因子语义问题 |
| 🟡 潜在 | 5 | GIL 限制、模块加载线程安全、代理超时、CSV 格式、进度心跳 |
| 🟢 优化 | 6 | 全选交互、截面阈值、滑块闪烁、API 性能、冗余导入、年化参数 |

**最需优先修复**的是 **#1 前瞻偏差**（ensemble_backtest 中 `shift(-1)`），因为它直接影响回测结果的可信度。

          
## 截面评估功能检查报告

经过全面检查，我发现了以下问题：

---

### 🔴 **严重问题：未来数据泄露风险**

#### 问题1：`calculate_factor_decay` 方法的逻辑错误

**位置**: [factor_evaluator.py:286-313](file:///d:\PythonProject\FactorMiner\factor_miner\core\factor_evaluator.py#L286-L313)

**问题描述**: 
代码注释声称"修复了未来函数问题"，但实际上修复后的逻辑仍然不正确：

```python
for lag in range(max_lag + 1):
    if lag == 0:
        ic = self.calculate_ic(factor, returns)
    else:
        lagged_factor = factor.shift(lag)  # ❌ 错误：使用历史因子值
        ic = self.calculate_ic(lagged_factor, returns)
```

**正确逻辑应该是**:
- `lag=0`: 因子值预测当期收益
- `lag=1`: 因子值预测1期后的收益 (factor vs returns.shift(-1))
- `lag=2`: 因子值预测2期后的收益 (factor vs returns.shift(-2))

**当前代码实际做的是**:
- `lag=1`: 用 t-1 时刻的因子预测 t 时刻收益
- `lag=2`: 用 t-2 时刻的因子预测 t 时刻收益

这**不是**因子衰减分析的正确含义。因子衰减应该测试**同一时刻的因子值**对不同期限未来收益的预测能力。

**修复建议**:
```python
for lag in range(max_lag + 1):
    if lag == 0:
        ic = self.calculate_ic(factor, returns)
    else:
        lagged_returns = returns.shift(-lag)  # 未来收益
        ic = self.calculate_ic(factor, lagged_returns)
```

⚠️ **注意**: 这会使用未来数据，但这是因子衰减分析的**正确做法**，因为目的是评估因子对不同期限收益的预测能力。关键是**只在评估阶段使用**，不能在实盘交易中使用。

---

### 🟡 **中等问题：逻辑异常**

#### 问题2：`calculate_factor_returns` 中 Sharpe 比率计算口径不一致

**位置**: [factor_evaluator.py:215-229](file:///d:\PythonProject\FactorMiner\factor_minuator.py#L215-L229)

**问题描述**:
```python
long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]
returns_std = returns.std()  # ❌ 使用原始收益的标准差
sharpe_ratio = long_short_return / returns_std
```

分子是多空收益差，分母却用原始收益率的标准差，**口径不一致**。

**修复建议**:
```python
long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]
# 使用多空收益序列的标准差
ls_series = returns.where(long_mask, 0.0) - returns.where(short_mask, 0.0)
ls_std = ls_series.std()
sharpe_ratio = long_short_return / ls_std if ls_std > 0 else np.nan
```

---

#### 问题3：`_resolve_signal_shift` 中 lookahead 处理逻辑可能有问题

**位置**: [factor_evaluator.py:818-860](file:///d:\PythonProject\FactorMiner\factor_miner\core\factor_evaluator.py#L818-L860)

**问题描述**:
```python
shift_n = int(shift_n) + int(lookahead)
```

如果因子定义中声明了 `lookahead`（表示因子使用了未来 N 根K线的数据），代码会额外延迟因子值。但这个逻辑的**语义不清晰**：
- 如果因子本身使用了未来数据，应该**禁止使用**或**标记为无效**
- 而不是简单地延迟因子值

**建议**: 明确文档说明 `lookahead` 参数的用途，或者直接拒绝使用包含未来数据的因子。

---

### 🟢 **已正确处理的地方**

1. **`prepare_cross_sectional_data` 中的时序对齐** ✅
   - `future_returns` 计算正确
   - `trade_shift` 默认为 1，确保因子值延迟一期使用

2. **`calculate_cross_sectional_ic` 中 NaN 处理** ✅
   - 正确处理了 `ic_std` 为 NaN 时 `icir` 也应为 NaN 的情况

3. **`calculate_cross_sectional_returns` 中分组赋值** ✅
   - 使用 `group.index` 避免了 RangeIndex 错位问题

---

### 🔵 **优化建议**

#### 优化1：添加数据泄露检测警告

建议在 `prepare_cross_sectional_data` 中添加检查：

```python
if trade_shift < 1:
    warnings.warn(
        f"trade_shift={trade_shift} 可能导致未来数据泄露！"
        f"因子值和未来收益在同一时间点对齐。",
        UserWarning
    )
```

#### 优化2：统一时序对齐策略

当前代码中存在多处 `shift(1)` 的硬编码：
- [comprehensive_factor_analysis:398](file:///d:\PythonProject\FactorMiner\factor_miner\core\factor_evaluator.py#L398)
- [calculate_factor_effectiveness_score:590](file:///d:\PythonProject\FactorMiner\factor_miner\core\factor_evaluator.py#L590)

建议统一使用配置参数或方法参数来控制延迟期数。

#### 优化3：添加单元测试验证时序正确性

建议添加测试用例验证：
1. 因子值和收益率的时序对齐是否正确
2. 是否存在未来数据泄露
3. 边界情况处理（如数据不足、全 NaN 等）

---

### 📋 **问题汇总表**

| 严重程度 | 问题 | 位置 | 状态 |
|---------|------|------|------|
| 🔴 高 | `calculate_factor_decay` 逻辑错误 | L286-313 | 需修复 |
| 🟡 中 | Sharpe 比率口径不一致 | L215-229 | 需修复 |
| 🟡 中 | `lookahead` 处理语义不清 | L818-860 | 需明确 |
| 🔵 低 | 缺少数据泄露警告 | - | 建议添加 |
| 🔵 低 | 时序对齐策略不统一 | 多处 | 建议优化 |

---

是否需要我帮你修复这些问题？
        
            
toolName: todo_write
            
status: success
          
Todos updated: 4 items
          
检查完成！主要发现了一个**严重的逻辑错误**（`calculate_factor_decay` 方法）和一个**中等的计算口径问题**（Sharpe 比率）。核心的截面评估数据准备逻辑是正确的，不存在未来数据泄露问题。