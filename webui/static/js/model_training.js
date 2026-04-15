let localDataRows = [];
let trainingSessionId = null;
let statusCheckInterval = null;
let featureImportanceChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    loadFactors();
    loadTrainingHistory();
});

function initializePage() {
    bindEventListeners();
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
}

async function refreshLocalData(forceRefresh) {
    const exchange = document.getElementById('exchangeSelect').value;
    const tradeType = document.getElementById('tradeTypeSelect').value;
    const cacheKey = `localData_${exchange}_${tradeType}`;

    if (!forceRefresh) {
        const cached = localStorage.getItem(cacheKey);
        if (cached) {
            try {
                const parsed = JSON.parse(cached);
                if (parsed.timestamp && Date.now() - parsed.timestamp < 300000) {
                    localDataRows = parsed.data || [];
                    updateSymbolSelect();
                    return;
                }
            } catch (e) {}
        }
    }

    const btn = document.getElementById('refreshDataBtn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>刷新中';
    }

    try {
        const forceParam = forceRefresh ? '&force=1' : '';
        const resp = await fetch(`/api/data/local-data-cached?exchange=${exchange}&trade_type=${tradeType}${forceParam}`);
        if (resp.ok) {
            const data = await resp.json();
            localDataRows = data.data || [];
            localStorage.setItem(cacheKey, JSON.stringify({
                data: localDataRows,
                timestamp: Date.now()
            }));
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
        updateRangeSlider(start, end);
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

function updateRangeSlider(startStr, endStr) {
    const startDateStr = startStr.split(' ')[0].split('T')[0];
    const endDateStr = endStr.split(' ')[0].split('T')[0];
    const startTs = parseDateTimestamp(startDateStr);
    const endTs = parseDateTimestamp(endDateStr);

    document.getElementById('startDate').value = startDateStr;
    document.getElementById('endDate').value = endDateStr;
    document.getElementById('rangeInfo').textContent = `${startDateStr} ~ ${endDateStr}`;

    const sliderEl = document.getElementById('rangeSlider');
    const fallbackEl = document.getElementById('dateRangeFallback');
    const startInput = document.getElementById('startDateInput');
    const endInput = document.getElementById('endDateInput');
    sliderEl.style.display = 'block';
    fallbackEl.style.display = 'none';
    sliderEl.style.display = 'none';
    fallbackEl.style.display = 'block';

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

function onDateInputChange() {
    const startInput = document.getElementById('startDateInput');
    const endInput = document.getElementById('endDateInput');
    let start = startInput.value;
    let end = endInput.value;

    if (start && end && start > end) {
        if (this && this.id === 'startDateInput') {
            end = start;
            endInput.value = end;
        } else {
            start = end;
            startInput.value = start;
        }
    }

    if (start) document.getElementById('startDate').value = start;
    if (end) document.getElementById('endDate').value = end;

    const displayStart = start || '-';
    const displayEnd = end || '-';
    document.getElementById('rangeInfo').textContent = `${displayStart} ~ ${displayEnd}`;
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
        'extra_data': {
            name: '额外数据因子',
            icon: 'fa-database',
            color: 'warning',
            desc: '需要资金费率、持仓量等额外数据',
            factors: []
        },
        'ml_pretrained': {
            name: 'ML预训练因子',
            icon: 'fa-brain',
            color: 'info',
            desc: '需要ML模型或预训练数据',
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
                        <div><strong>${subGroup.name}</strong> (${subGroup.factors.length})</div>
                        <div class="form-check m-0">
                            <input class="form-check-input" type="checkbox"
                                   id="subgroup_toggle_${key}_${subKey}"
                                   onchange="toggleSubgroupSelection('${key}_${subKey}', this.checked)">
                            <label class="form-check-label" for="subgroup_toggle_${key}_${subKey}">全选</label>
                        </div>
                    </div>
                    ${subGroup.factors.map(f => `
                        <div class="form-check factor-item" data-factor-id="${f.id}" data-subgroup="${key}_${subKey}">
                            <input class="form-check-input factor-checkbox" type="checkbox"
                                   value="${f.id}" id="factor_${f.id}"
                                   onchange="updateFactorCount()">
                            <label class="form-check-label small" for="factor_${f.id}"
                                   title="${f.description || ''}">
                                ${f.name}
                                <span class="text-muted">(${f.type || ''})</span>
                            </label>
                        </div>
                    `).join('')}
                    <hr class="my-2">
                `;
            });
        } else {
            factorsHtml = cat.factors.map(f => `
                <div class="form-check factor-item" data-factor-id="${f.id}" data-category="${key}">
                    <input class="form-check-input factor-checkbox" type="checkbox"
                           value="${f.id}" id="factor_${f.id}"
                           onchange="updateFactorCount()">
                    <label class="form-check-label small" for="factor_${f.id}"
                           title="${f.description || ''}">
                        ${f.name}
                        <span class="text-muted">(${f.type || ''})</span>
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
                        <strong>${cat.name}</strong>
                        <span class="badge bg-secondary ms-2">${cat.factors.length}</span>
                        <small class="text-muted ms-2 d-none d-sm-inline">${cat.desc}</small>
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

function onModelTypeChange() {
    const modelType = document.getElementById('modelTypeSelect').value;
    const container = document.getElementById('modelParamsContainer');

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
        container.innerHTML = `
            <div class="row">
                <div class="col-md-6 mb-3">
                    <label class="form-label">正则化强度 C / alpha</label>
                    <input type="number" class="form-control" id="paramC" value="1.0" min="0.01" max="100" step="0.1">
                    <small class="text-muted">分类用C，回归用alpha</small>
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
}

function onTaskTypeChange() {
    onLossFunctionChange();
}

function onLabelTypeChange() {
    const labelType = document.getElementById('labelTypeSelect').value;
    const taskTypeSel = document.getElementById('taskTypeSelect');

    if (labelType === 'direction') {
        taskTypeSel.value = 'classification';
        taskTypeSel.disabled = true;
    } else {
        taskTypeSel.disabled = false;
    }
    onLossFunctionChange();
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
        if (isClassification) {
            opt.disabled = val !== 'log_loss';
        } else {
            opt.disabled = val === 'log_loss';
        }
    }

    if (isClassification) {
        if (lossSel.value !== 'log_loss') {
            lossSel.value = 'log_loss';
        }
        paramsContainer.style.display = 'none';
    } else {
        if (lossSel.value === 'log_loss') {
            lossSel.value = 'mse';
        }
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
        const c = document.getElementById('paramC');
        const mi = document.getElementById('paramMaxIter');
        if (c) params.C = parseFloat(c.value) || 1.0;
        if (c) params.alpha = parseFloat(c.value) || 1.0;
        if (mi) params.max_iter = parseInt(mi.value) || 1000;
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

async function startTraining() {
    const filePath = document.getElementById('filePath').value;
    if (!filePath) {
        alert('请先选择数据');
        return;
    }

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

    const btn = document.getElementById('startTrainingBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>训练中...';

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
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play me-2"></i>开始训练';
        }
    } catch (e) {
        alert('请求失败: ' + e.message);
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play me-2"></i>开始训练';
    }
}

function startStatusPolling() {
    if (statusCheckInterval) clearInterval(statusCheckInterval);
    statusCheckInterval = setInterval(checkTrainingStatus, 1500);
}

async function checkTrainingStatus() {
    if (!trainingSessionId) return;

    try {
        const resp = await fetch(`/api/training/status/${trainingSessionId}`);
        const data = await resp.json();

        if (!data.success) return;

        const pct = data.progress || 0;
        document.getElementById('overallProgress').style.width = pct + '%';
        document.getElementById('progressPercent').textContent = pct + '%';
        document.getElementById('stepMessage').textContent = data.message || '';

        const badge = document.getElementById('progressBadge');
        badge.textContent = data.status === 'running' ? '训练中' : data.status === 'completed' ? '完成' : data.status === 'failed' ? '失败' : '准备中';
        badge.className = 'badge ms-2 ' + (
            data.status === 'running' ? 'bg-primary' :
            data.status === 'completed' ? 'bg-success' :
            data.status === 'failed' ? 'bg-danger' : 'bg-secondary'
        );

        if (data.status === 'completed') {
            clearInterval(statusCheckInterval);
            await showTrainingResults(trainingSessionId);
            const btn = document.getElementById('startTrainingBtn');
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play me-2"></i>开始训练';
            loadTrainingHistory();
        } else if (data.status === 'failed') {
            clearInterval(statusCheckInterval);
            alert('训练失败: ' + (data.message || '未知错误'));
            const btn = document.getElementById('startTrainingBtn');
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play me-2"></i>开始训练';
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
                            <h4>${(metrics.val?.accuracy * 100 || 0).toFixed(1)}%</h4>
                            <p class="mb-0">验证集准确率</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-success text-white text-center">
                        <div class="card-body">
                            <h4>${(metrics.val?.f1 || 0).toFixed(4)}</h4>
                            <p class="mb-0">验证集F1</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white text-center">
                        <div class="card-body">
                            <h4>${(metrics.val?.auc || 0).toFixed(4)}</h4>
                            <p class="mb-0">验证集AUC</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-white text-center">
                        <div class="card-body">
                            <h4>${dataInfo.total_samples || 0}</h4>
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
                            <h4>${(metrics.val?.rmse || 0).toFixed(6)}</h4>
                            <p class="mb-0">验证集RMSE</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-success text-white text-center">
                        <div class="card-body">
                            <h4>${(metrics.val?.direction_accuracy * 100 || 0).toFixed(1)}%</h4>
                            <p class="mb-0">验证集方向准确率</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white text-center">
                        <div class="card-body">
                            <h4>${(metrics.val?.r2 || 0).toFixed(4)}</h4>
                            <p class="mb-0">验证集R²</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-white text-center">
                        <div class="card-body">
                            <h4>${dataInfo.total_samples || 0}</h4>
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
            ['训练集', `${dataInfo.train_samples} (${dataInfo.train_range})`],
            ['测试集', `${dataInfo.test_samples} (${dataInfo.test_range})`],
            ['验证集', `${dataInfo.val_samples} (${dataInfo.val_range})`],
            ['特征数', dataInfo.feature_count],
            ['标签', dataInfo.label_name],
        ];
        infoRows.forEach(([k, v]) => {
            dataInfoBody.innerHTML += `<tr><td>${k}</td><td>${v || '-'}</td></tr>`;
        });

        renderFeatureImportance(featureImportance);
    } catch (e) {
        console.error('显示结果失败:', e);
    }
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
            if (key === 'direction_accuracy' || key === 'accuracy' || key === 'precision' || key === 'recall') {
                displayVal = (val * 100).toFixed(2) + '%';
            } else {
                displayVal = val.toFixed(6);
            }
        } else {
            displayVal = val;
        }
        tbody.innerHTML += `<tr><td>${name}</td><td>${displayVal}</td></tr>`;
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
    try {
        const resp = await fetch('/api/training/history');
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.success) return;

        const tbody = document.getElementById('historyTable');
        const history = data.history || [];

        if (history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">暂无训练历史</td></tr>';
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
                coreMetric = m.accuracy ? (m.accuracy * 100).toFixed(1) + '%' : '-';
            } else {
                coreMetric = m.direction_accuracy ? (m.direction_accuracy * 100).toFixed(1) + '%' : (m.rmse ? m.rmse.toFixed(6) : '-');
            }

            tbody.innerHTML += `
                <tr>
                    <td><small>${h.model_id || '-'}</small></td>
                    <td>${modelNames[h.model_type] || h.model_type}</td>
                    <td>${labelNames[h.label_type] || h.label_type}</td>
                    <td>${h.predict_step || '-'}</td>
                    <td>${lossNames[h.loss_function] || h.loss_function}</td>
                    <td>${coreMetric}</td>
                    <td><small>${h.created_at ? h.created_at.slice(0, 19) : '-'}</small></td>
                    <td>
                        <button class="btn btn-outline-danger btn-sm" onclick="deleteModel('${h.model_id}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        });
    } catch (e) {
        console.error('加载训练历史失败:', e);
    }
}

async function deleteModel(modelId) {
    if (!confirm(`确定删除模型 ${modelId}？`)) return;
    try {
        const resp = await fetch(`/api/training/delete/${modelId}`, { method: 'DELETE' });
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
