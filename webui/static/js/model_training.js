let localDataRows = [];
let trainingSessionId = null;
let statusCheckInterval = null;
let featureImportanceChart = null;
let trainingCurveChart = null;
let cachedModels = [];
let cachedLossFunctions = {};
let cachedLabelTypes = [];
let predictLocalDataRows = [];
let historySearchDebounceTimer = null;

function safeNumber(value, fallback) {
    return Number.isFinite(value) ? value : (fallback !== undefined ? fallback : 0);
}

function formatPercent(value, digits) {
    if (!Number.isFinite(value)) return '-';
    return (value * 100).toFixed(digits !== undefined ? digits : 2) + '%';
}

function formatNumber(value, digits) {
    if (!Number.isFinite(value)) return '-';
    return value.toFixed(digits !== undefined ? digits : 6);
}

document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    loadFactors();
    loadTrainingHistory();
});

async function initializePage() {
    bindEventListeners();
    await Promise.all([
        loadModelsFromApi(),
        loadLossFunctionsFromApi(),
        loadLabelTypesFromApi(),
    ]);
    onSplitChange();
    onModelTypeChange();
    onLabelTypeChange();
    onLossFunctionChange();
    setTimeout(() => refreshLocalData(), 200);
}

function bindEventListeners() {
    document.getElementById('exchangeSelect').addEventListener('change', refreshLocalData);
    document.getElementById('tradeTypeSelect').addEventListener('change', refreshLocalData);
    document.getElementById('symbolSelect').addEventListener('change', onSymbolChange);
    document.getElementById('timeframeSelect').addEventListener('change', onTimeframeChange);
    document.getElementById('factorSearch').addEventListener('input', filterFactors);
    document.getElementById('startDateInput').addEventListener('change', onDateInputChange);
    document.getElementById('endDateInput').addEventListener('change', onDateInputChange);

    const historySearchEl = document.getElementById('historySearch');
    if (historySearchEl) {
        historySearchEl.addEventListener('input', () => {
            if (historySearchDebounceTimer) clearTimeout(historySearchDebounceTimer);
            historySearchDebounceTimer = setTimeout(loadTrainingHistory, 300);
        });
    }

    const predictExchange = document.getElementById('predictExchangeSelect');
    const predictTradeType = document.getElementById('predictTradeTypeSelect');
    const predictSymbol = document.getElementById('predictSymbolSelect');
    const predictTimeframe = document.getElementById('predictTimeframeSelect');
    if (predictExchange && predictTradeType && predictSymbol && predictTimeframe) {
        predictExchange.addEventListener('change', refreshPredictLocalData);
        predictTradeType.addEventListener('change', refreshPredictLocalData);
        predictSymbol.addEventListener('change', onPredictSymbolChange);
        predictTimeframe.addEventListener('change', onPredictTimeframeChange);
    }
}

async function loadModelsFromApi() {
    try {
        const resp = await fetch('/api/training/models');
        if (resp.ok) {
            const data = await resp.json();
            if (data.success) {
                cachedModels = data.models || [];
                const sel = document.getElementById('modelTypeSelect');
                const currentVal = sel.value;
                sel.innerHTML = '';
                cachedModels.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.id;
                    opt.textContent = m.name;
                    opt.title = m.description || '';
                    sel.appendChild(opt);
                });
                if (currentVal && cachedModels.some(m => m.id === currentVal)) {
                    sel.value = currentVal;
                }
            }
        }
    } catch (e) {
        console.error('加载模型列表失败:', e);
    }
}

async function loadLossFunctionsFromApi() {
    try {
        const resp = await fetch('/api/training/loss-functions');
        if (resp.ok) {
            const data = await resp.json();
            if (data.success) {
                cachedLossFunctions = data.loss_functions || {};
                const sel = document.getElementById('lossFunctionSelect');
                const currentVal = sel.value;
                sel.innerHTML = '';
                Object.entries(cachedLossFunctions).forEach(([key, lf]) => {
                    const opt = document.createElement('option');
                    opt.value = key;
                    opt.textContent = lf.name + (lf.type ? ` (${lf.type === 'classification' ? '分类' : '回归'})` : '');
                    opt.title = lf.description || '';
                    sel.appendChild(opt);
                });
                if (currentVal && cachedLossFunctions[currentVal]) {
                    sel.value = currentVal;
                }
            }
        }
    } catch (e) {
        console.error('加载损失函数列表失败:', e);
    }
}

async function loadLabelTypesFromApi() {
    try {
        const resp = await fetch('/api/training/label-types');
        if (resp.ok) {
            const data = await resp.json();
            if (data.success) {
                cachedLabelTypes = data.label_types || [];
                const sel = document.getElementById('labelTypeSelect');
                const currentVal = sel.value;
                sel.innerHTML = '';
                cachedLabelTypes.forEach(lt => {
                    const opt = document.createElement('option');
                    opt.value = lt.id;
                    opt.textContent = lt.name;
                    opt.title = lt.description || '';
                    sel.appendChild(opt);
                });
                if (currentVal && cachedLabelTypes.some(lt => lt.id === currentVal)) {
                    sel.value = currentVal;
                }
            }
        }
    } catch (e) {
        console.error('加载标签类型列表失败:', e);
    }
}

async function refreshLocalData(forceRefresh) {
    const exchange = document.getElementById('exchangeSelect').value;
    const tradeType = document.getElementById('tradeTypeSelect').value;

    const btn = document.getElementById('refreshDataBtn');
    if (btn && forceRefresh) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>刷新中';
    }

    try {
        const forceParam = forceRefresh ? '&force=1' : '';
        const resp = await fetch(`/api/data/local-data-cached?exchange=${exchange}&trade_type=${tradeType}${forceParam}`);
        if (resp.ok) {
            const data = await resp.json();
            localDataRows = data.data || [];
            updateSymbolSelect();
        }
    } catch (e) {
        console.error('获取本地数据失败:', e);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-sync-alt me-1"></i>刷新数据';
        }
    }
}

function updateSymbolSelect() {
    const sel = document.getElementById('symbolSelect');
    const symbols = [...new Set(localDataRows.map(r => r.symbol))].sort();
    sel.innerHTML = '<option value="">请选择...</option>';
    symbols.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;
        sel.appendChild(opt);
    });
}

function onSymbolChange() {
    const symbol = document.getElementById('symbolSelect').value;
    const tfSel = document.getElementById('timeframeSelect');

    const rows = localDataRows.filter(r => r.symbol === symbol);
    const timeframes = [...new Set(rows.map(r => r.timeframe))].sort((a, b) => timeframeToMinutes(a) - timeframeToMinutes(b));

    tfSel.innerHTML = '<option value="">请选择...</option>';
    timeframes.forEach(tf => {
        const opt = document.createElement('option');
        opt.value = tf;
        opt.textContent = tf;
        tfSel.appendChild(opt);
    });

    if (timeframes.length === 1) {
        tfSel.value = timeframes[0];
        onTimeframeChange();
    }
}

function onTimeframeChange() {
    const symbol = document.getElementById('symbolSelect').value;
    const timeframe = document.getElementById('timeframeSelect').value;
    if (!symbol || !timeframe) return;

    const row = localDataRows.find(r => r.symbol === symbol && r.timeframe === timeframe);
    if (!row) {
        console.warn('未找到匹配数据:', symbol, timeframe, '可用行数:', localDataRows.length);
        return;
    }

    document.getElementById('filePath').value = row.file_path;

    const start = row.date_range?.start;
    const end = row.date_range?.end;
    if (start && end) {
        updateDateRange(start, end);
    } else {
        console.warn('日期范围缺失:', row);
    }
}

function normalizeDateString(dateStr) {
    if (!dateStr || typeof dateStr !== 'string') {
        return '';
    }
    return dateStr.trim().replace(' ', 'T');
}

function parseDateTimestamp(dateStr) {
    const normalized = normalizeDateString(dateStr);
    if (!normalized) {
        return NaN;
    }
    const ts = Date.parse(normalized);
    return Number.isFinite(ts) ? ts : NaN;
}

function updateDateRange(startStr, endStr) {
    const startDateStr = startStr.split(' ')[0].split('T')[0];
    const endDateStr = endStr.split(' ')[0].split('T')[0];
    const startTs = parseDateTimestamp(startDateStr);
    const endTs = parseDateTimestamp(endDateStr);

    document.getElementById('startDate').value = startDateStr;
    document.getElementById('endDate').value = endDateStr;
    document.getElementById('rangeInfo').textContent = `${startDateStr} ~ ${endDateStr}`;

    const startInput = document.getElementById('startDateInput');
    const endInput = document.getElementById('endDateInput');

    if (Number.isFinite(startTs) && Number.isFinite(endTs) && startTs <= endTs) {
        startInput.min = startDateStr;
        startInput.max = endDateStr;
        endInput.min = startDateStr;
        endInput.max = endDateStr;
        startInput.value = startDateStr;
        endInput.value = endDateStr;
    } else {
        startInput.min = '';
        startInput.max = '';
        endInput.min = '';
        endInput.max = '';
        startInput.value = startDateStr || '';
        endInput.value = endDateStr || '';
    }
}

function onDateInputChange(event) {
    const startInput = document.getElementById('startDateInput');
    const endInput = document.getElementById('endDateInput');
    let start = startInput.value;
    let end = endInput.value;

    const rangeInfoEl = document.getElementById('rangeInfo');
    if (rangeInfoEl) rangeInfoEl.classList.remove('text-danger');

    if (start && end && start > end) {
        const triggerId = (event && event.target) ? event.target.id : (this && this.id);
        if (triggerId === 'startDateInput') {
            end = start;
            endInput.value = end;
        } else {
            start = end;
            startInput.value = start;
        }
        if (rangeInfoEl) {
            rangeInfoEl.classList.add('text-danger');
            setTimeout(() => rangeInfoEl.classList.remove('text-danger'), 1500);
        }
    }

    if (start) document.getElementById('startDate').value = start;
    if (end) document.getElementById('endDate').value = end;

    const displayStart = start || '-';
    const displayEnd = end || '-';
    if (rangeInfoEl) rangeInfoEl.textContent = `${displayStart} ~ ${displayEnd}`;
}

function timeframeToMinutes(tf) {
    const unit = tf.slice(-1);
    const val = parseInt(tf.slice(0, -1));
    switch (unit) {
        case 'm': return val;
        case 'h': return val * 60;
        case 'd': return val * 1440;
        default: return 0;
    }
}

let allFactors = [];

async function loadFactors() {
    try {
        const resp = await fetch('/api/factors/list');
        if (resp.ok) {
            const data = await resp.json();
            allFactors = data.factors || [];
            renderFactorList(allFactors);
        }
    } catch (e) {
        console.error('加载因子列表失败:', e);
    }
}

function isWindowBasedKlineFactor(factor) {
    const factorId = factor.id || '';
    const windowPatterns = ['ma_', 'ema_', 'sma_', 'rsi_', 'macd', 'atr_', 'std_', 'var_', 
                            'max_', 'min_', 'sum_', 'roc_', 'momentum', 'adx', 'cci', 
                            'willr', 'stoch', 'bb_', 'kc_', 'dc_', 'obv', 'mfi', 
                            '_ma', '_ema', '_sma', 'volatility', 'trend'];
    for (const p of windowPatterns) {
        if (factorId.toLowerCase().includes(p)) return true;
    }
    if (/(^|_)\d{1,3}($|_)/.test(factorId)) return true;
    return false;
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function renderFactorList(factors) {
    const container = document.getElementById('factorListContainer');
    
    const categories = {
        'basic_kline': {
            name: '基础K线因子',
            icon: 'fa-chart-line',
            color: 'primary',
            desc: '仅需OHLCV数据即可计算',
            factors: [],
            subgroups: {
                'single_bar': { name: '单bar因子', factors: [] },
                'window': { name: '窗口因子（MA/RSI等）', factors: [] }
            }
        },
        'event_factor': {
            name: '事件因子',
            icon: 'fa-bolt',
            color: 'danger',
            desc: '方向信号/交叉突破等离散事件类因子',
            factors: []
        },
        'derivatives': {
            name: '衍生品微观结构因子',
            icon: 'fa-database',
            color: 'warning',
            desc: '需要持仓量、多空比、主动买入量、基差等衍生品数据',
            factors: []
        },
        'funding': {
            name: '资金费率因子',
            icon: 'fa-coins',
            color: 'info',
            desc: '需要资金费率（Funding Rate）历史',
            factors: []
        }
    };
    
    factors.forEach(f => {
        const req = f.data_requirement || 'basic_kline';
        if (categories[req]) {
            categories[req].factors.push(f);
            if (req === 'basic_kline') {
                const subgroup = isWindowBasedKlineFactor(f) ? 'window' : 'single_bar';
                categories[req].subgroups[subgroup].factors.push(f);
            }
        } else {
            categories['basic_kline'].factors.push(f);
            categories['basic_kline'].subgroups['single_bar'].factors.push(f);
        }
    });
    
    let html = '';
    Object.entries(categories).forEach(([key, cat]) => {
        if (cat.factors.length === 0) return;
        
        const collapseId = `collapse_${key}`;
        let factorsHtml = '';
        
        if (key === 'basic_kline' && cat.subgroups) {
            Object.entries(cat.subgroups).forEach(([subKey, subGroup]) => {
                if (subGroup.factors.length === 0) return;
                factorsHtml += `
                    <div class="subgroup-header d-flex justify-content-between align-items-center small text-muted mb-1 px-1">
                        <div><strong>${escapeHtml(subGroup.name)}</strong> (${subGroup.factors.length})</div>
                        <div class="form-check m-0">
                            <input class="form-check-input" type="checkbox"
                                   id="subgroup_toggle_${key}_${subKey}"
                                   onchange="toggleSubgroupSelection('${key}_${subKey}', this.checked)">
                            <label class="form-check-label" for="subgroup_toggle_${key}_${subKey}">全选</label>
                        </div>
                    </div>
                    ${subGroup.factors.map(f => `
                        <div class="form-check factor-item" data-factor-id="${escapeHtml(f.id)}" data-subgroup="${key}_${subKey}">
                            <input class="form-check-input factor-checkbox" type="checkbox"
                                   value="${escapeHtml(f.id)}" id="factor_${escapeHtml(f.id)}"
                                   onchange="updateFactorCount()">
                            <label class="form-check-label small" for="factor_${escapeHtml(f.id)}"
                                   title="${escapeHtml(f.description || '')}">
                                ${escapeHtml(f.name)}
                                <span class="text-muted">(${escapeHtml(f.type || '')})</span>
                            </label>
                        </div>
                    `).join('')}
                    <hr class="my-2">
                `;
            });
        } else {
            factorsHtml = cat.factors.map(f => `
                <div class="form-check factor-item" data-factor-id="${escapeHtml(f.id)}" data-category="${key}">
                    <input class="form-check-input factor-checkbox" type="checkbox"
                           value="${escapeHtml(f.id)}" id="factor_${escapeHtml(f.id)}"
                           onchange="updateFactorCount()">
                    <label class="form-check-label small" for="factor_${escapeHtml(f.id)}"
                           title="${escapeHtml(f.description || '')}">
                        ${escapeHtml(f.name)}
                        <span class="text-muted">(${escapeHtml(f.type || '')})</span>
                    </label>
                </div>
            `).join('');
        }
        
        html += `
            <div class="factor-category">
                <div class="d-flex justify-content-between align-items-center p-2 bg-light border-bottom"
                     data-bs-toggle="collapse" data-bs-target="#${collapseId}" style="cursor: pointer;">
                    <div>
                        <i class="fas ${cat.icon} text-${cat.color} me-2"></i>
                        <strong>${escapeHtml(cat.name)}</strong>
                        <span class="badge bg-secondary ms-2">${cat.factors.length}</span>
                        <small class="text-muted ms-2 d-none d-sm-inline">${escapeHtml(cat.desc)}</small>
                    </div>
                    <div class="d-flex align-items-center">
                        <button type="button" class="btn btn-link btn-sm p-0 me-2"
                                onclick="event.stopPropagation(); toggleCategorySelection('${key}');">
                            <i class="fas fa-check-square"></i>
                        </button>
                        <i class="fas fa-chevron-down collapse-icon"></i>
                    </div>
                </div>
                <div id="${collapseId}" class="collapse show">
                    <div class="p-2">
                        ${factorsHtml}
                    </div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html || '<div class="text-center text-muted p-3">暂无因子</div>';
    document.getElementById('totalFactorCount').textContent = factors.length;
    updateFactorCount();
}

function toggleSubgroupSelection(subgroupId, checked) {
    const items = document.querySelectorAll(`.factor-item[data-subgroup="${subgroupId}"] .factor-checkbox`);
    items.forEach(cb => cb.checked = checked);
    updateFactorCount();
}

function toggleCategorySelection(categoryKey) {
    const checkboxes = document.querySelectorAll(`#collapse_${categoryKey} .factor-checkbox`);
    const allChecked = Array.from(checkboxes).every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
    updateFactorCount();
}

function updateFactorCount() {
    const checked = document.querySelectorAll('.factor-checkbox:checked').length;
    document.getElementById('selectedFactorCount').textContent = checked;
}

function filterFactors() {
    const keyword = document.getElementById('factorSearch').value.toLowerCase().trim();
    const items = document.querySelectorAll('.factor-item');
    
    items.forEach(item => {
        const label = item.querySelector('label').textContent.toLowerCase();
        item.style.display = keyword && !label.includes(keyword) ? 'none' : '';
    });
}

function selectAllFactors() {
    document.querySelectorAll('.factor-checkbox').forEach(cb => cb.checked = true);
    updateFactorCount();
}

function clearFactorSelection() {
    document.querySelectorAll('.factor-checkbox').forEach(cb => cb.checked = false);
    updateFactorCount();
}

function getSelectedFactorIds() {
    return Array.from(document.querySelectorAll('.factor-checkbox:checked')).map(cb => cb.value);
}

const MODEL_PARAM_CACHE = {};

function _snapshotCurrentParams() {
    const container = document.getElementById('modelParamsContainer');
    if (!container) return;
    const inputs = container.querySelectorAll('input[id]');
    const snapshot = {};
    inputs.forEach(input => {
        snapshot[input.id] = input.value;
    });
    const prevModelType = container.getAttribute('data-model-type');
    if (prevModelType) {
        MODEL_PARAM_CACHE[prevModelType] = Object.assign(
            MODEL_PARAM_CACHE[prevModelType] || {},
            snapshot
        );
    }
}

function _restoreParams(modelType) {
    const cache = MODEL_PARAM_CACHE[modelType] || {};
    Object.entries(cache).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el != null && value !== undefined && value !== null && value !== '') {
            el.value = value;
        }
    });
}

function onModelTypeChange() {
    _snapshotCurrentParams();
    const modelType = document.getElementById('modelTypeSelect').value;
    const container = document.getElementById('modelParamsContainer');
    container.setAttribute('data-model-type', modelType);

    if (modelType === 'lightgbm' || modelType === 'xgboost') {
        container.innerHTML = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">迭代次数 (n_estimators)</label>
                    <input type="number" class="form-control" id="paramNEstimators" value="200" min="10" max="5000">
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">学习率 (learning_rate)</label>
                    <input type="number" class="form-control" id="paramLearningRate" value="0.05" min="0.001" max="1" step="0.01">
                </div>
            </div>
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">最大深度 (max_depth)</label>
                    <input type="number" class="form-control" id="paramMaxDepth" value="-1" min="-1" max="50">
                    <small class="text-muted">-1表示不限制</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">Early Stopping轮数</label>
                    <input type="number" class="form-control" id="paramEarlyStopping" value="20" min="5" max="100">
                </div>
            </div>
        `;
    } else if (modelType === 'logistic_regression') {
        const taskType = document.getElementById('taskTypeSelect').value;
        const isCls = taskType === 'classification';
        container.innerHTML = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">${isCls ? '正则化强度 (C)' : '正则化强度 (alpha)'}</label>
                    <input type="number" class="form-control" id="${isCls ? 'paramC' : 'paramAlpha'}" value="1.0" min="0.01" max="100" step="0.1">
                    <small class="text-muted">${isCls ? '分类: LogisticRegression C' : '回归: Ridge alpha'}</small>
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">最大迭代次数</label>
                    <input type="number" class="form-control" id="paramMaxIter" value="1000" min="100" max="10000">
                </div>
            </div>
        `;
    } else if (modelType === 'random_forest') {
        container.innerHTML = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">树数量 (n_estimators)</label>
                    <input type="number" class="form-control" id="paramNEstimators" value="200" min="10" max="5000">
                </div>
                <div class="col-md-6 mb-3">
                    <label class="form-label">最大深度 (max_depth)</label>
                    <input type="number" class="form-control" id="paramMaxDepth" value="" min="1" max="50">
                    <small class="text-muted">留空表示不限制</small>
                </div>
            </div>
        `;
    }

    _restoreParams(modelType);
}

function onTaskTypeChange() {
    const modelType = document.getElementById('modelTypeSelect').value;
    if (modelType === 'logistic_regression') {
        onModelTypeChange();
    }
    onLossFunctionChange();
}

function onLabelTypeChange() {
    const labelType = document.getElementById('labelTypeSelect').value;
    const taskTypeSel = document.getElementById('taskTypeSelect');

    const ltInfo = cachedLabelTypes.find(lt => lt.id === labelType);
    if (ltInfo && ltInfo.task === 'classification') {
        taskTypeSel.value = 'classification';
        taskTypeSel.disabled = true;
    } else {
        taskTypeSel.disabled = false;
    }
    onTaskTypeChange();
}

function onLossFunctionChange() {
    const taskType = document.getElementById('taskTypeSelect').value;
    const labelType = document.getElementById('labelTypeSelect').value;
    const lossSel = document.getElementById('lossFunctionSelect');
    const paramsContainer = document.getElementById('lossParamsContainer');
    const paramsContent = document.getElementById('lossParamsContent');

    const isClassification = taskType === 'classification' || labelType === 'direction';

    for (const opt of lossSel.options) {
        const val = opt.value;
        const lfInfo = cachedLossFunctions[val];
        if (isClassification) {
            opt.disabled = !lfInfo || lfInfo.type !== 'classification';
        } else {
            opt.disabled = lfInfo && lfInfo.type === 'classification';
        }
    }

    const hint = document.getElementById('lossFunctionHint');
    if (isClassification) {
        const previous = lossSel.value;
        if (previous !== 'log_loss') {
            lossSel.value = 'log_loss';
            if (hint) {
                hint.textContent = '分类任务已自动切换为"对数损失"';
                hint.style.display = 'inline';
            }
        } else if (hint) {
            hint.style.display = 'none';
        }
        paramsContainer.style.display = 'none';
    } else {
        if (lossSel.value === 'log_loss') {
            lossSel.value = 'mse';
        }
        const hintEl = document.getElementById('lossFunctionHint');
        if (hintEl) hintEl.style.display = 'none';
        const lossFn = lossSel.value;
        if (lossFn === 'direction_aware_mse') {
            paramsContainer.style.display = 'block';
            paramsContent.innerHTML = `
                <div class="mb-2">
                    <label class="form-label small">λ (方向错误放大系数)</label>
                    <input type="number" class="form-control form-control-sm" id="lossLambda" value="2.0" min="0.5" max="10" step="0.5">
                </div>
            `;
        } else if (lossFn === 'composite') {
            paramsContainer.style.display = 'block';
            paramsContent.innerHTML = `
                <div class="mb-2">
                    <label class="form-label small">α (MSE权重)</label>
                    <input type="number" class="form-control form-control-sm" id="lossAlpha" value="1.0" min="0" max="5" step="0.1">
                </div>
                <div class="mb-2">
                    <label class="form-label small">β (方向损失权重)</label>
                    <input type="number" class="form-control form-control-sm" id="lossBeta" value="1.0" min="0" max="5" step="0.1">
                </div>
                <div class="mb-2">
                    <label class="form-label small">k (tanh陡峭度)</label>
                    <input type="number" class="form-control form-control-sm" id="lossK" value="5.0" min="1" max="20" step="1">
                </div>
            `;
        } else if (lossFn === 'magnitude_weighted') {
            paramsContainer.style.display = 'block';
            paramsContent.innerHTML = `
                <div class="mb-2">
                    <label class="form-label small">λ (幅度加权系数)</label>
                    <input type="number" class="form-control form-control-sm" id="lossLambda" value="2.0" min="0.5" max="10" step="0.5">
                </div>
            `;
        } else {
            paramsContainer.style.display = 'none';
        }
    }
}

function onSplitChange() {
    const trainR = parseFloat(document.getElementById('trainRatio').value) || 0.8;
    const testR = parseFloat(document.getElementById('testRatio').value) || 0.1;
    const valR = parseFloat(document.getElementById('valRatio').value) || 0.1;
    const total = trainR + testR + valR;

    const trainPct = (trainR / total * 100).toFixed(1);
    const testPct = (testR / total * 100).toFixed(1);
    const valPct = (valR / total * 100).toFixed(1);

    document.getElementById('trainBar').style.width = trainPct + '%';
    document.getElementById('trainBar').textContent = `训练 ${trainPct}%`;
    document.getElementById('testBar').style.width = testPct + '%';
    document.getElementById('testBar').textContent = `测试 ${testPct}%`;
    document.getElementById('valBar').style.width = valPct + '%';
    document.getElementById('valBar').textContent = `验证 ${valPct}%`;

    const warning = document.getElementById('splitWarning');
    if (Math.abs(total - 1.0) > 0.01) {
        warning.textContent = `⚠️ 比例之和为 ${total.toFixed(2)}，将自动归一化`;
        warning.className = 'text-warning';
    } else {
        warning.textContent = '';
    }
}

function getModelParams() {
    const modelType = document.getElementById('modelTypeSelect').value;
    const taskType = document.getElementById('taskTypeSelect').value;
    const params = {};

    if (modelType === 'lightgbm' || modelType === 'xgboost') {
        const nEst = document.getElementById('paramNEstimators');
        const lr = document.getElementById('paramLearningRate');
        const md = document.getElementById('paramMaxDepth');
        const es = document.getElementById('paramEarlyStopping');
        if (nEst) params.n_estimators = parseInt(nEst.value) || 200;
        if (lr) params.learning_rate = parseFloat(lr.value) || 0.05;
        if (md) params.max_depth = parseInt(md.value) || -1;
        if (es) params.early_stopping_rounds = parseInt(es.value) || 20;
    } else if (modelType === 'logistic_regression') {
        const mi = document.getElementById('paramMaxIter');
        if (mi) params.max_iter = parseInt(mi.value) || 1000;
        if (taskType === 'classification') {
            const c = document.getElementById('paramC');
            if (c) params.C = parseFloat(c.value) || 1.0;
        } else {
            const alpha = document.getElementById('paramAlpha');
            if (alpha) params.alpha = parseFloat(alpha.value) || 1.0;
        }
    } else if (modelType === 'random_forest') {
        const nEst = document.getElementById('paramNEstimators');
        const md = document.getElementById('paramMaxDepth');
        if (nEst) params.n_estimators = parseInt(nEst.value) || 200;
        if (md && md.value) params.max_depth = parseInt(md.value);
    }

    return params;
}

function getLossParams() {
    const lossFn = document.getElementById('lossFunctionSelect').value;
    const params = {};

    if (lossFn === 'direction_aware_mse') {
        const el = document.getElementById('lossLambda');
        params.lambda = el ? parseFloat(el.value) || 2.0 : 2.0;
    } else if (lossFn === 'composite') {
        const alpha = document.getElementById('lossAlpha');
        const beta = document.getElementById('lossBeta');
        const k = document.getElementById('lossK');
        params.alpha = alpha ? parseFloat(alpha.value) || 1.0 : 1.0;
        params.beta = beta ? parseFloat(beta.value) || 1.0 : 1.0;
        params.k = k ? parseFloat(k.value) || 5.0 : 5.0;
    } else if (lossFn === 'magnitude_weighted') {
        const el = document.getElementById('lossLambda');
        params.lambda = el ? parseFloat(el.value) || 2.0 : 2.0;
    }

    return params;
}

function validateTrainingInputs() {
    const filePath = document.getElementById('filePath').value;
    if (!filePath) {
        alert('请先选择数据');
        return false;
    }

    if (!filePath.toLowerCase().endsWith('.feather')) {
        alert('数据文件必须是 .feather 格式');
        return false;
    }

    const predictStep = parseInt(document.getElementById('predictStep').value);
    if (!predictStep || predictStep < 1 || predictStep > 100) {
        alert('预测步长必须为1-100之间的正整数');
        return false;
    }

    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    if (startDate && endDate && startDate > endDate) {
        alert('开始日期不能晚于结束日期');
        return false;
    }

    const trainR = parseFloat(document.getElementById('trainRatio').value) || 0;
    const testR = parseFloat(document.getElementById('testRatio').value) || 0;
    const valR = parseFloat(document.getElementById('valRatio').value) || 0;
    if (trainR <= 0 || testR <= 0 || valR <= 0) {
        alert('训练/测试/验证集比例必须都大于0');
        return false;
    }
    const splitTotal = trainR + testR + valR;
    if (splitTotal > 3.0 || splitTotal < 0.3) {
        alert('数据集划分比例异常（总和应接近 1），请检查输入');
        return false;
    }

    const modelType = document.getElementById('modelTypeSelect').value;
    const modelParams = getModelParams();

    if (modelType === 'lightgbm' || modelType === 'xgboost') {
        if (modelParams.n_estimators && (modelParams.n_estimators < 10 || modelParams.n_estimators > 5000)) {
            alert('迭代次数必须在10-5000之间');
            return false;
        }
        if (modelParams.learning_rate && (modelParams.learning_rate < 0.001 || modelParams.learning_rate > 1)) {
            alert('学习率必须在0.001-1之间');
            return false;
        }
        if (modelParams.early_stopping_rounds && (modelParams.early_stopping_rounds < 5 || modelParams.early_stopping_rounds > 100)) {
            alert('Early Stopping轮数必须在5-100之间');
            return false;
        }
    } else if (modelType === 'logistic_regression') {
        if (modelParams.max_iter && (modelParams.max_iter < 100 || modelParams.max_iter > 10000)) {
            alert('最大迭代次数必须在100-10000之间');
            return false;
        }
    } else if (modelType === 'random_forest') {
        if (modelParams.n_estimators && (modelParams.n_estimators < 10 || modelParams.n_estimators > 5000)) {
            alert('树数量必须在10-5000之间');
            return false;
        }
    }

    const lossFn = document.getElementById('lossFunctionSelect').value;
    const taskType = document.getElementById('taskTypeSelect').value;
    const labelType = document.getElementById('labelTypeSelect').value;
    const isClassification = taskType === 'classification' || labelType === 'direction';
    if (isClassification && lossFn !== 'log_loss') {
        alert('分类任务仅支持对数损失函数');
        return false;
    }

    return true;
}

function _resetTrainingButtons() {
    const startBtn = document.getElementById('startTrainingBtn');
    const cancelBtn = document.getElementById('cancelTrainingBtn');
    if (startBtn) {
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-play me-2"></i>开始训练';
    }
    if (cancelBtn) {
        cancelBtn.style.display = 'none';
        cancelBtn.disabled = false;
        cancelBtn.innerHTML = '<i class="fas fa-stop me-2"></i>取消训练';
    }
}

function _enterTrainingUI() {
    const startBtn = document.getElementById('startTrainingBtn');
    const cancelBtn = document.getElementById('cancelTrainingBtn');
    startBtn.disabled = true;
    startBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>训练中...';
    if (cancelBtn) {
        cancelBtn.style.display = '';
        cancelBtn.disabled = false;
    }
    const failedAlert = document.getElementById('failedFactorsAlert');
    if (failedAlert) failedAlert.style.display = 'none';
}

async function startTraining() {
    if (!validateTrainingInputs()) return;

    const filePath = document.getElementById('filePath').value;
    const factorIds = getSelectedFactorIds();

    const trainR = parseFloat(document.getElementById('trainRatio').value) || 0.8;
    const testR = parseFloat(document.getElementById('testRatio').value) || 0.1;
    const valR = parseFloat(document.getElementById('valRatio').value) || 0.1;
    const total = trainR + testR + valR;

    const payload = {
        file_path: filePath,
        start_date: document.getElementById('startDate').value || null,
        end_date: document.getElementById('endDate').value || null,
        factor_ids: factorIds,
        label_type: document.getElementById('labelTypeSelect').value,
        predict_step: parseInt(document.getElementById('predictStep').value) || 1,
        model_type: document.getElementById('modelTypeSelect').value,
        task_type: document.getElementById('taskTypeSelect').value,
        loss_function: document.getElementById('lossFunctionSelect').value,
        train_ratio: trainR / total,
        test_ratio: testR / total,
        val_ratio: valR / total,
        model_params: getModelParams(),
        loss_params: getLossParams(),
    };

    _enterTrainingUI();

    document.getElementById('trainingProgressSection').style.display = 'block';
    document.getElementById('trainingResultsSection').style.display = 'none';

    try {
        const resp = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();

        if (data.success) {
            trainingSessionId = data.session_id;
            startStatusPolling();
        } else {
            alert('启动训练失败: ' + (data.error || '未知错误'));
            _resetTrainingButtons();
        }
    } catch (e) {
        alert('请求失败: ' + e.message);
        _resetTrainingButtons();
    }
}

async function cancelTraining() {
    if (!trainingSessionId) return;
    const cancelBtn = document.getElementById('cancelTrainingBtn');
    if (cancelBtn) {
        cancelBtn.disabled = true;
        cancelBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>取消中...';
    }
    try {
        const resp = await fetch(`/api/training/cancel/${trainingSessionId}`, { method: 'POST' });
        const data = await resp.json();
        if (!data.success) {
            alert('取消失败: ' + (data.error || '未知错误'));
            if (cancelBtn) {
                cancelBtn.disabled = false;
                cancelBtn.innerHTML = '<i class="fas fa-stop me-2"></i>取消训练';
            }
        }
    } catch (e) {
        alert('取消请求失败: ' + e.message);
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.innerHTML = '<i class="fas fa-stop me-2"></i>取消训练';
        }
    }
}

function startStatusPolling() {
    if (statusCheckInterval) clearInterval(statusCheckInterval);
    statusCheckInterval = setInterval(checkTrainingStatus, 1500);
}

function _renderFailedFactorsAlert(failedFactors) {
    const alertEl = document.getElementById('failedFactorsAlert');
    const textEl = document.getElementById('failedFactorsText');
    if (!alertEl || !textEl) return;
    if (!failedFactors || failedFactors.length === 0) {
        alertEl.style.display = 'none';
        return;
    }
    const preview = failedFactors.slice(0, 5).map(f =>
        `${escapeHtml(f.factor_id)}${f.reason ? ` (${escapeHtml(String(f.reason).slice(0, 80))})` : ''}`
    ).join('; ');
    const more = failedFactors.length > 5 ? ` ... 共 ${failedFactors.length} 个` : '';
    textEl.innerHTML = preview + more;
    alertEl.style.display = '';
}

async function checkTrainingStatus() {
    if (!trainingSessionId) return;

    try {
        const resp = await fetch(`/api/training/status/${trainingSessionId}`);
        const data = await resp.json();

        if (!data.success) return;

        const pct = safeNumber(data.progress, 0);
        document.getElementById('overallProgress').style.width = pct + '%';
        document.getElementById('progressPercent').textContent = pct + '%';
        document.getElementById('stepMessage').textContent = data.message || '';

        _renderFailedFactorsAlert(data.failed_factors);

        const badge = document.getElementById('progressBadge');
        const statusTextMap = {
            running: '训练中', completed: '完成', failed: '失败',
            cancelled: '已取消', pending: '准备中',
        };
        badge.textContent = statusTextMap[data.status] || '准备中';
        const badgeClassMap = {
            running: 'bg-primary', completed: 'bg-success',
            failed: 'bg-danger', cancelled: 'bg-secondary',
            pending: 'bg-secondary',
        };
        badge.className = 'badge ms-2 ' + (badgeClassMap[data.status] || 'bg-secondary');

        if (data.status === 'completed') {
            clearInterval(statusCheckInterval);
            await showTrainingResults(trainingSessionId);
            _resetTrainingButtons();
            loadTrainingHistory();
        } else if (data.status === 'failed') {
            clearInterval(statusCheckInterval);
            alert('训练失败: ' + (data.message || '未知错误'));
            _resetTrainingButtons();
        } else if (data.status === 'cancelled') {
            clearInterval(statusCheckInterval);
            _resetTrainingButtons();
        }
    } catch (e) {
        console.error('状态检查失败:', e);
    }
}

async function showTrainingResults(sessionId) {
    try {
        const resp = await fetch(`/api/training/result/${sessionId}`);
        const data = await resp.json();
        if (!data.success) {
            alert('获取结果失败: ' + data.error);
            return;
        }

        const results = data.results;
        displayResultsData(results);

        if (results.evals_result) {
            renderTrainingCurve(results.evals_result, results.model_type);
        }
    } catch (e) {
        console.error('显示结果失败:', e);
    }
}

function displayResultsData(results) {
    const metrics = results.metrics || {};
    const dataInfo = results.data_info || {};
    const featureImportance = results.feature_importance || {};

    document.getElementById('trainingResultsSection').style.display = 'block';

    const overview = document.getElementById('metricsOverview');
    const isClassification = results.task_type === 'classification' || results.label_type === 'direction';

    if (isClassification) {
        overview.innerHTML = `
            <div class="col-md-3">
                <div class="card bg-primary text-white text-center">
                    <div class="card-body">
                        <h4>${formatPercent(metrics.val?.accuracy, 1)}</h4>
                        <p class="mb-0">验证集准确率</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-success text-white text-center">
                    <div class="card-body">
                        <h4>${formatNumber(metrics.val?.f1, 4)}</h4>
                        <p class="mb-0">验证集F1</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white text-center">
                    <div class="card-body">
                        <h4>${formatNumber(metrics.val?.auc, 4)}</h4>
                        <p class="mb-0">验证集AUC</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white text-center">
                    <div class="card-body">
                        <h4>${safeNumber(dataInfo.total_samples, 0)}</h4>
                        <p class="mb-0">总样本数</p>
                    </div>
                </div>
            </div>
        `;
    } else {
        overview.innerHTML = `
            <div class="col-md-3">
                <div class="card bg-primary text-white text-center">
                    <div class="card-body">
                        <h4>${formatNumber(metrics.val?.rmse, 6)}</h4>
                        <p class="mb-0">验证集RMSE</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-success text-white text-center">
                    <div class="card-body">
                        <h4>${formatPercent(metrics.val?.direction_accuracy, 1)}</h4>
                        <p class="mb-0">验证集方向准确率</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-info text-white text-center">
                    <div class="card-body">
                        <h4>${formatNumber(metrics.val?.r2, 4)}</h4>
                        <p class="mb-0">验证集R²</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-warning text-white text-center">
                    <div class="card-body">
                        <h4>${safeNumber(dataInfo.total_samples, 0)}</h4>
                        <p class="mb-0">总样本数</p>
                    </div>
                </div>
            </div>
        `;
    }

    fillMetricsTable('testMetricsTable', metrics.test || {});
    fillMetricsTable('valMetricsTable', metrics.val || {});

    const dataInfoBody = document.querySelector('#dataInfoTable tbody');
    dataInfoBody.innerHTML = '';
    const infoRows = [
        ['总样本数', dataInfo.total_samples],
        ['训练集 (早 → train)', `${dataInfo.train_samples} (${dataInfo.train_range})`],
        ['验证集 (中 → val, early stopping)', `${dataInfo.val_samples} (${dataInfo.val_range})`],
        ['测试集 (晚 → test, 最终评估)', `${dataInfo.test_samples} (${dataInfo.test_range})`],
        ['特征数', dataInfo.feature_count],
        ['标签', dataInfo.label_name],
    ];
    infoRows.forEach(([k, v]) => {
        dataInfoBody.innerHTML += `<tr><td>${escapeHtml(String(k))}</td><td>${escapeHtml(String(v || '-'))}</td></tr>`;
    });

    renderFeatureImportance(featureImportance);
}

const CURVE_COLORS = [
    'rgba(54, 162, 235, 0.8)',
    'rgba(255, 99, 132, 0.8)',
    'rgba(75, 192, 192, 0.8)',
    'rgba(255, 159, 64, 0.8)',
    'rgba(153, 102, 255, 0.8)',
    'rgba(201, 203, 207, 0.8)',
];

function _buildCurveDatasets(evalsResult) {
    const datasets = [];
    let colorIdx = 0;
    for (const [datasetName, datasetMetrics] of Object.entries(evalsResult || {})) {
        for (const [metricName, values] of Object.entries(datasetMetrics || {})) {
            datasets.push({
                label: `${datasetName}/${metricName}`,
                data: Array.isArray(values) ? values : [],
                borderColor: CURVE_COLORS[colorIdx % CURVE_COLORS.length],
                backgroundColor: CURVE_COLORS[colorIdx % CURVE_COLORS.length].replace('0.8', '0.1'),
                fill: false,
                tension: 0.1,
                pointRadius: 0,
            });
            colorIdx++;
        }
    }
    return datasets;
}

function _renderCurveChart(evalsResult, sectionId, canvasId, existingChart) {
    const curveSection = document.getElementById(sectionId);
    if (!curveSection) return existingChart;

    const datasets = _buildCurveDatasets(evalsResult);
    if (datasets.length === 0) {
        curveSection.style.display = 'none';
        if (existingChart) { try { existingChart.destroy(); } catch (_) {} }
        return null;
    }
    curveSection.style.display = 'block';

    if (existingChart) { try { existingChart.destroy(); } catch (_) {} }

    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const maxLen = datasets.reduce((m, d) => Math.max(m, d.data.length), 0);
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels: Array.from({ length: maxLen }, (_, i) => i + 1),
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: '训练评估曲线' },
            },
            scales: {
                x: { title: { display: true, text: '迭代轮次' } },
                y: { title: { display: true, text: '评估指标值' } },
            },
        },
    });
}

function renderTrainingCurve(evalsResult, modelType) {
    trainingCurveChart = _renderCurveChart(
        evalsResult,
        'trainingCurveSection',
        'trainingCurveChart',
        trainingCurveChart
    );
}

let detailCurveChart = null;

function renderDetailCurve(evalsResult, modelType) {
    detailCurveChart = _renderCurveChart(
        evalsResult,
        'detailCurveSection',
        'detailCurveChart',
        detailCurveChart
    );
}

function fillMetricsTable(tableId, metrics) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = '';
    const metricNames = {
        'mse': 'MSE',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R²',
        'direction_accuracy': '方向准确率',
        'direction_aware_mse': '方向感知MSE',
        'composite_loss': '复合损失',
        'magnitude_weighted_loss': '幅度加权损失',
        'accuracy': '准确率',
        'precision': '精确率',
        'recall': '召回率',
        'f1': 'F1',
        'auc': 'AUC',
        'log_loss': '对数损失',
    };
    for (const [key, val] of Object.entries(metrics)) {
        const name = metricNames[key] || key;
        let displayVal;
        if (typeof val === 'number') {
            if (!Number.isFinite(val)) {
                displayVal = '-';
            } else if (key === 'direction_accuracy' || key === 'accuracy' || key === 'precision' || key === 'recall') {
                displayVal = (val * 100).toFixed(2) + '%';
            } else {
                displayVal = val.toFixed(6);
            }
        } else {
            displayVal = val == null ? '-' : String(val);
        }
        tbody.innerHTML += `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(displayVal)}</td></tr>`;
    }
}

function renderFeatureImportance(importance) {
    const topN = Object.entries(importance).slice(0, 15);
    if (topN.length === 0) return;

    const labels = topN.map(([k]) => k.length > 20 ? k.slice(0, 18) + '...' : k);
    const values = topN.map(([, v]) => v);

    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }

    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '重要性',
                data: values,
                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { beginAtZero: true }
            }
        }
    });
}

async function loadTrainingHistory() {
    const tbody = document.getElementById('historyTable');
    const totalBadge = document.getElementById('historyTotalBadge');
    const searchEl = document.getElementById('historySearch');
    const search = searchEl ? searchEl.value.trim() : '';

    try {
        const params = new URLSearchParams();
        if (search) params.set('search', search);
        const qs = params.toString();
        const resp = await fetch(`/api/training/history${qs ? '?' + qs : ''}`);
        if (!resp.ok) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-danger">加载失败，请点击右上角刷新按钮重试</td></tr>';
            return;
        }
        const data = await resp.json();
        if (!data.success) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center text-danger">加载失败: ${escapeHtml(data.error || '未知错误')}</td></tr>`;
            return;
        }

        const history = data.history || [];
        if (totalBadge) totalBadge.textContent = data.total != null ? `(共 ${data.total})` : '';

        if (history.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted">${search ? '无匹配结果' : '暂无训练历史'}</td></tr>`;
            return;
        }

        tbody.innerHTML = '';
        history.forEach(h => {
            const labelNames = { 'log_return': '对数收益率', 'direction': '收益方向', 'composite': '综合' };
            const modelNames = { 'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'logistic_regression': '逻辑回归', 'random_forest': '随机森林' };
            const lossNames = { 'mse': 'MSE', 'direction_aware_mse': '方向感知MSE', 'composite': '复合损失', 'magnitude_weighted': '幅度加权', 'log_loss': '对数损失' };

            let coreMetric = '-';
            const m = h.metrics?.val || {};
            if (h.label_type === 'direction') {
                coreMetric = Number.isFinite(m.accuracy) ? (m.accuracy * 100).toFixed(1) + '%' : '-';
            } else if (Number.isFinite(m.direction_accuracy)) {
                coreMetric = (m.direction_accuracy * 100).toFixed(1) + '%';
            } else if (Number.isFinite(m.rmse)) {
                coreMetric = m.rmse.toFixed(6);
            }

            const safeModelId = escapeHtml(h.model_id || '-');
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><small>${safeModelId}</small></td>
                <td>${escapeHtml(modelNames[h.model_type] || h.model_type)}</td>
                <td>${escapeHtml(labelNames[h.label_type] || h.label_type)}</td>
                <td>${h.predict_step || '-'}</td>
                <td>${escapeHtml(lossNames[h.loss_function] || h.loss_function)}</td>
                <td>${escapeHtml(coreMetric)}</td>
                <td><small>${escapeHtml(h.created_at ? h.created_at.slice(0, 19) : '-')}</small></td>
                <td>
                    <button class="btn btn-outline-info btn-sm me-1" onclick="viewModelDetail('${safeModelId}')" title="查看详情">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn btn-outline-primary btn-sm me-1" onclick="openPredictModal('${safeModelId}')" title="预测">
                        <i class="fas fa-chart-line"></i>
                    </button>
                    <button class="btn btn-outline-danger btn-sm" onclick="deleteModel('${safeModelId}')" title="删除">
                        <i class="fas fa-trash"></i>
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error('加载训练历史失败:', e);
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="8" class="text-center text-danger">加载失败: ${escapeHtml(e.message || '网络错误')}</td></tr>`;
        }
    }
}

async function viewModelDetail(modelId) {
    try {
        const resp = await fetch(`/api/training/detail/${encodeURIComponent(modelId)}`);
        const data = await resp.json();
        if (!data.success) {
            alert('获取详情失败: ' + data.error);
            return;
        }

        const d = data.detail;
        const isClassification = d.task_type === 'classification' || d.label_type === 'direction';
        const metrics = d.metrics || {};
        const dataInfo = d.data_info || {};

        const labelNames = { 'log_return': '对数收益率', 'direction': '收益方向', 'composite': '综合' };
        const modelNames = { 'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'logistic_regression': '逻辑回归', 'random_forest': '随机森林' };
        const lossNames = { 'mse': 'MSE', 'direction_aware_mse': '方向感知MSE', 'composite': '复合损失', 'magnitude_weighted': '幅度加权', 'log_loss': '对数损失' };

        let html = `
            <div class="row mb-3">
                <div class="col-md-6">
                    <h6>基本信息</h6>
                    <table class="table table-sm">
                        <tr><td>模型ID</td><td>${escapeHtml(d.model_id)}</td></tr>
                        <tr><td>模型类型</td><td>${escapeHtml(modelNames[d.model_type] || d.model_type)}</td></tr>
                        <tr><td>任务类型</td><td>${d.task_type === 'classification' ? '分类' : '回归'}</td></tr>
                        <tr><td>标签类型</td><td>${escapeHtml(labelNames[d.label_type] || d.label_type)}</td></tr>
                        <tr><td>预测步长</td><td>${d.predict_step}</td></tr>
                        <tr><td>损失函数</td><td>${escapeHtml(lossNames[d.loss_function] || d.loss_function)}</td></tr>
                        <tr><td>创建时间</td><td>${escapeHtml(d.created_at || '-')}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>数据信息</h6>
                    <table class="table table-sm">
                        <tr><td>总样本数</td><td>${dataInfo.total_samples || '-'}</td></tr>
                        <tr><td>训练集</td><td>${dataInfo.train_samples || '-'} (${dataInfo.train_range || '-'})</td></tr>
                        <tr><td>验证集 (early stopping)</td><td>${dataInfo.val_samples || '-'} (${dataInfo.val_range || '-'})</td></tr>
                        <tr><td>测试集 (最终评估)</td><td>${dataInfo.test_samples || '-'} (${dataInfo.test_range || '-'})</td></tr>
                        <tr><td>特征数</td><td>${dataInfo.feature_count || '-'}</td></tr>
                        <tr><td>标签</td><td>${escapeHtml(dataInfo.label_name || '-')}</td></tr>
                        <tr><td>划分比例</td><td>训练${formatNumber(d.train_ratio, 2)} / 验证${formatNumber(d.val_ratio, 2)} / 测试${formatNumber(d.test_ratio, 2)}</td></tr>
                        <tr><td>数据文件</td><td><small class="text-muted" title="${escapeHtml(d.data_file || '')}">${escapeHtml((d.data_file || '-').split(/[\\/]/).pop())}</small></td></tr>
                        <tr><td>日期范围</td><td><small>${escapeHtml(d.start_date || '起始')} ~ ${escapeHtml(d.end_date || '结束')}</small></td></tr>
                    </table>
                </div>
            </div>
        `;

        if (d.loss_function_note) {
            html += `<div class="alert alert-warning py-2 mb-3"><small>${escapeHtml(d.loss_function_note)}</small></div>`;
        }

        html += `<h6>评估指标</h6><div class="row"><div class="col-md-6"><small class="text-muted">测试集</small>`;
        html += buildMetricList(metrics.test || {});
        html += `</div><div class="col-md-6"><small class="text-muted">验证集</small>`;
        html += buildMetricList(metrics.val || {});
        html += `</div></div>`;

        if (d.feature_names && d.feature_names.length > 0) {
            html += `<h6 class="mt-3">特征列表 (${d.feature_names.length})</h6>`;
            html += `<div style="max-height:150px;overflow-y:auto;"><small>${d.feature_names.map(f => escapeHtml(f)).join(', ')}</small></div>`;
        }

        if (d.factor_ids && d.factor_ids.length > 0) {
            html += `<h6 class="mt-3">使用的因子 (${d.factor_ids.length})</h6>`;
            html += `<div style="max-height:120px;overflow-y:auto;"><small>${d.factor_ids.map(f => escapeHtml(f)).join(', ')}</small></div>`;
        }

        if (d.failed_factors && d.failed_factors.length > 0) {
            const failedHtml = d.failed_factors.map(item =>
                `<li><code>${escapeHtml(item.factor_id)}</code>: ${escapeHtml(String(item.reason || '').slice(0, 160))}</li>`
            ).join('');
            html += `<div class="alert alert-warning mt-3 py-2 mb-0"><small><strong>部分因子计算失败 (${d.failed_factors.length}):</strong><ul class="mb-0">${failedHtml}</ul></small></div>`;
        }

        html += `<div id="detailCurveSection" class="mt-3" style="display:none;">
            <h6>训练评估曲线</h6>
            <div style="height: 300px;">
                <canvas id="detailCurveChart"></canvas>
            </div>
        </div>`;

        document.getElementById('detailModalBody').innerHTML = html;

        const curveResp = await fetch(`/api/training/training-curve/${encodeURIComponent(modelId)}`);
        const curveData = await curveResp.json();
        if (curveData.success && curveData.curve) {
            renderDetailCurve(curveData.curve, curveData.model_type);
        }

        const modal = new bootstrap.Modal(document.getElementById('detailModal'));
        modal.show();
    } catch (e) {
        alert('获取详情失败: ' + e.message);
    }
}

function buildMetricList(metrics) {
    const metricNames = {
        'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R²',
        'direction_accuracy': '方向准确率', 'direction_aware_mse': '方向感知MSE',
        'composite_loss': '复合损失', 'magnitude_weighted_loss': '幅度加权损失',
        'accuracy': '准确率', 'precision': '精确率', 'recall': '召回率',
        'f1': 'F1', 'auc': 'AUC', 'log_loss': '对数损失',
    };
    let html = '<table class="table table-sm"><tbody>';
    for (const [key, val] of Object.entries(metrics)) {
        const name = metricNames[key] || key;
        let displayVal;
        if (typeof val === 'number') {
            if (!Number.isFinite(val)) {
                displayVal = '-';
            } else if (['direction_accuracy', 'accuracy', 'precision', 'recall'].includes(key)) {
                displayVal = (val * 100).toFixed(2) + '%';
            } else {
                displayVal = val.toFixed(6);
            }
        } else {
            displayVal = val == null ? '-' : String(val);
        }
        html += `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(displayVal)}</td></tr>`;
    }
    html += '</tbody></table>';
    return html;
}

function openPredictModal(modelId) {
    document.getElementById('predictModelId').value = modelId;
    document.getElementById('predictResultSection').style.display = 'none';
    const filePathInput = document.getElementById('predictFilePath');
    if (filePathInput) filePathInput.value = '';
    refreshPredictLocalData();
    const modal = new bootstrap.Modal(document.getElementById('predictModal'));
    modal.show();
}

async function refreshPredictLocalData() {
    const ex = document.getElementById('predictExchangeSelect');
    const tt = document.getElementById('predictTradeTypeSelect');
    if (!ex || !tt) return;
    try {
        const resp = await fetch(`/api/data/local-data-cached?exchange=${ex.value}&trade_type=${tt.value}`);
        if (!resp.ok) return;
        const data = await resp.json();
        predictLocalDataRows = data.data || [];
        const symbolSel = document.getElementById('predictSymbolSelect');
        if (symbolSel) {
            const symbols = [...new Set(predictLocalDataRows.map(r => r.symbol))].sort();
            symbolSel.innerHTML = '<option value="">请选择...</option>';
            symbols.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s; opt.textContent = s;
                symbolSel.appendChild(opt);
            });
        }
        const tfSel = document.getElementById('predictTimeframeSelect');
        if (tfSel) tfSel.innerHTML = '<option value="">请选择...</option>';
    } catch (e) {
        console.error('加载预测数据列表失败:', e);
    }
}

function onPredictSymbolChange() {
    const symbol = document.getElementById('predictSymbolSelect').value;
    const tfSel = document.getElementById('predictTimeframeSelect');
    const rows = predictLocalDataRows.filter(r => r.symbol === symbol);
    const timeframes = [...new Set(rows.map(r => r.timeframe))]
        .sort((a, b) => timeframeToMinutes(a) - timeframeToMinutes(b));
    tfSel.innerHTML = '<option value="">请选择...</option>';
    timeframes.forEach(tf => {
        const opt = document.createElement('option');
        opt.value = tf; opt.textContent = tf;
        tfSel.appendChild(opt);
    });
    if (timeframes.length === 1) {
        tfSel.value = timeframes[0];
        onPredictTimeframeChange();
    }
}

function onPredictTimeframeChange() {
    const symbol = document.getElementById('predictSymbolSelect').value;
    const timeframe = document.getElementById('predictTimeframeSelect').value;
    if (!symbol || !timeframe) return;
    const row = predictLocalDataRows.find(r => r.symbol === symbol && r.timeframe === timeframe);
    if (!row) return;
    const fpInput = document.getElementById('predictFilePath');
    if (fpInput) fpInput.value = row.file_path;
}

async function runPrediction() {
    const modelId = document.getElementById('predictModelId').value;
    const filePath = document.getElementById('predictFilePath').value;
    const startDate = document.getElementById('predictStartDate').value;
    const endDate = document.getElementById('predictEndDate').value;

    if (!filePath) {
        alert('请输入数据文件路径');
        return;
    }

    const btn = document.getElementById('runPredictBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>预测中...';

    try {
        const resp = await fetch(`/api/training/predict/${encodeURIComponent(modelId)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: filePath,
                start_date: startDate || null,
                end_date: endDate || null,
            }),
        });
        const data = await resp.json();

        if (data.success) {
            const predictions = data.predictions?.prediction || [];
            const index = data.index || [];
            const n = Math.min(predictions.length, 100);

            let html = `<p>共 ${predictions.length} 条预测结果，展示前 ${n} 条：</p>`;
            html += '<table class="table table-sm"><thead><tr><th>时间</th><th>预测值</th></tr></thead><tbody>';
            for (let i = 0; i < n; i++) {
                const v = predictions[i];
                const valText = Number.isFinite(v) ? v.toFixed(6) : '-';
                html += `<tr><td>${escapeHtml(index[i] || '-')}</td><td>${valText}</td></tr>`;
            }
            html += '</tbody></table>';

            document.getElementById('predictResultContent').innerHTML = html;
            document.getElementById('predictResultSection').style.display = 'block';
        } else {
            alert('预测失败: ' + (data.error || '未知错误'));
        }
    } catch (e) {
        alert('预测请求失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-chart-line me-1"></i>执行预测';
    }
}

async function deleteModel(modelId) {
    if (!confirm(`确定删除模型 ${modelId}？`)) return;
    try {
        const resp = await fetch(`/api/training/delete/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
        const data = await resp.json();
        if (data.success) {
            loadTrainingHistory();
        } else {
            alert('删除失败: ' + data.error);
        }
    } catch (e) {
        alert('删除失败: ' + e.message);
    }
}
