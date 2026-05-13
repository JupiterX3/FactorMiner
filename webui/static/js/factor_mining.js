/**
 * 因子挖掘页面JavaScript
 */

// 全局变量
let localDataRows = [];
let rangeSlider = null;
let miningSession = null;
let progressInterval = null;
let progressEventSource = null;
let currentMiningMode = 'standard';
let allSymbols = [];
let stablecoins = new Set();

// 步骤名称映射（后端步骤名称 -> 前端步骤ID）
const STEP_MAPPING = {
    'data_loading': 'step1',
    'factor_building': 'step2', 
    'factor_evaluation': 'step3',
    'factor_optimization': 'step4',
    'result_saving': 'step5'
};

// 步骤显示名称映射
const STEP_DISPLAY_NAMES = {
    'data_loading': '数据加载',
    'factor_building': '因子构建',
    'factor_evaluation': '因子评估',
    'factor_optimization': '因子优化',
    'result_saving': '结果保存'
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('因子挖掘页面初始化');
    initializePage();
    loadAlgorithms();
});

// 子进度统计（前端自估计）
let __mlSubStartTs = null;
let __mlLastPct = 0;
let __mlLastTs = 0;

/**
 * 初始化页面
 */
function initializePage() {
    console.log('开始初始化页面...');
    
    // 绑定事件监听器
    bindEventListeners();
    console.log('事件监听器绑定完成');
    
    // 初始化数据选择器
    initializeDataSelectors();
    console.log('数据选择器初始化完成');

    // 加载稳定币列表（用于交易对筛选）
    loadStablecoins();
    
    // 加载挖掘历史
    loadMiningHistory();
    console.log('挖掘历史加载完成');
    
    console.log('页面初始化完成');
}

/**
 * 绑定事件监听器
 */
function bindEventListeners() {
    // 表单提交
    const miningForm = document.getElementById('miningForm');
    if (miningForm) {
        miningForm.addEventListener('submit', handleMiningSubmit);
        console.log('挖掘表单事件监听器已绑定');
    } else {
        console.error('找不到挖掘表单元素');
    }

    // 兜底：截面模式下直接响应按钮点击，避免浏览器原生表单拦截导致“无反应”
    const startMiningBtn = document.getElementById('startMiningBtn');
    if (startMiningBtn) {
        startMiningBtn.addEventListener('click', async (event) => {
            if (currentMiningMode === 'cross_sectional') {
                event.preventDefault();
                await handleCrossSectionalMining();
            } else if (currentMiningMode === 'cross_sectional_rl') {
                event.preventDefault();
                await handleRLCrossSectionalMining();
            }
        });
    }
    
    // 数据选择器变化
    const exchangeSelect = document.getElementById('exchangeSelect');
    if (exchangeSelect) {
        exchangeSelect.addEventListener('change', refreshLocalMeta);
    }
    
    const tradeTypeSelect = document.getElementById('tradeTypeSelect');
    if (tradeTypeSelect) {
        tradeTypeSelect.addEventListener('change', refreshLocalMeta);
    }
    
    const symbolsSelect = document.getElementById('symbolsSelect');
    if (symbolsSelect) {
        symbolsSelect.addEventListener('change', () => {
            updateSelectedSymbolCount();
            updateTimeframesForSelection();
        });
    }
    
    const timeframesSelect = document.getElementById('timeframesSelect');
    if (timeframesSelect) {
        timeframesSelect.addEventListener('change', updateRangeForSelection);
    }

    const symbolSearchInput = document.getElementById('symbolSearchInput');
    if (symbolSearchInput) {
        symbolSearchInput.addEventListener('input', filterSymbolsByKeyword);
    }

    const excludeStablecoinsFilter = document.getElementById('excludeStablecoinsFilter');
    if (excludeStablecoinsFilter) {
        excludeStablecoinsFilter.addEventListener('change', filterSymbolsByStablecoin);
    }

    const startDateDisplay = document.getElementById('startDateDisplay');
    if (startDateDisplay) {
        startDateDisplay.addEventListener('change', onManualDateInputChanged);
    }

    const endDateDisplay = document.getElementById('endDateDisplay');
    if (endDateDisplay) {
        endDateDisplay.addEventListener('change', onManualDateInputChanged);
    }
}

/**
 * 初始化数据选择器
 */
function initializeDataSelectors() {
    console.log('初始化数据选择器...');
    
    // 初始化交易所选择器
    const exchangeSelect = document.getElementById('exchangeSelect');
    console.log('交易所选择器元素:', exchangeSelect);
    
    if (exchangeSelect) {
        exchangeSelect.innerHTML = '<option value="">请选择...</option>';
        exchangeSelect.innerHTML += '<option value="binance">Binance</option>';
        exchangeSelect.innerHTML += '<option value="okx">OKX</option>';
        exchangeSelect.innerHTML += '<option value="bybit">Bybit</option>';
        console.log('交易所选择器已初始化，选项数:', exchangeSelect.options.length);
    } else {
        console.error('找不到交易所选择器元素');
    }
    
    // 初始化交易类型选择器
    const tradeTypeSelect = document.getElementById('tradeTypeSelect');
    console.log('交易类型选择器元素:', tradeTypeSelect);
    
    if (tradeTypeSelect) {
        tradeTypeSelect.innerHTML = '<option value="">请选择...</option>';
        tradeTypeSelect.innerHTML += '<option value="futures">期货</option>';
        tradeTypeSelect.innerHTML += '<option value="spot">现货</option>';
        console.log('交易类型选择器已初始化，选项数:', tradeTypeSelect.options.length);
    } else {
        console.error('找不到交易类型选择器元素');
    }
    
    // 默认选择第一个选项
    if (exchangeSelect && exchangeSelect.options.length > 1) {
        exchangeSelect.selectedIndex = 1;
    }
    
    if (tradeTypeSelect && tradeTypeSelect.options.length > 1) {
        tradeTypeSelect.selectedIndex = 1;
    }
    
    // 延迟调用，确保DOM完全加载
    setTimeout(() => {
        if (exchangeSelect && exchangeSelect.value && tradeTypeSelect && tradeTypeSelect.value) {
            refreshLocalMeta();
        }
    }, 100);
}

/**
 * 刷新本地数据元信息
 */
async function refreshLocalMeta(forceRefresh = false) {
    const exchangeSelect = document.getElementById('exchangeSelect');
    const tradeTypeSelect = document.getElementById('tradeTypeSelect');
    
    console.log('refreshLocalMeta 被调用', { forceRefresh });
    console.log('交易所选择:', exchangeSelect?.value);
    console.log('交易类型选择:', tradeTypeSelect?.value);
    
    if (!exchangeSelect || !tradeTypeSelect) {
        console.error('找不到交易所或交易类型选择器');
        return;
    }
    
    if (!exchangeSelect.value || !tradeTypeSelect.value) {
        console.log('交易所或交易类型未选择，跳过刷新');
        return;
    }

    const btn = document.getElementById('refreshDataBtn');
    if (btn && forceRefresh) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-sync-alt fa-spin me-1"></i>刷新中...';
    }
    
    try {
        const forceParam = forceRefresh ? '&force=1' : '';
        console.log(`请求API: /api/data/local-data?exchange=${exchangeSelect.value}&trade_type=${tradeTypeSelect.value}${forceParam}`);
        const response = await fetch(`/api/data/local-data?exchange=${exchangeSelect.value}&trade_type=${tradeTypeSelect.value}${forceParam}`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('API返回数据:', data);
            localDataRows = data.data || [];
            console.log('解析后的数据行:', localDataRows);
            
            if (localDataRows.length > 0) {
                updateSymbolsSelect();
                updateTimeframesSelect();
                updateRangeForSelection();
                console.log('数据选择器更新完成');
                
                const cacheInfo = data.cached ? `缓存数据 (更新于 ${data.cached_at || '未知'})` : '已从磁盘刷新';
                console.log('数据来源:', cacheInfo);
            } else {
                console.log('没有找到数据');
                showAlert('warning', '没有找到可用的数据');
            }
        } else {
            console.error('API请求失败:', response.status, response.statusText);
            showAlert('error', `API请求失败: ${response.status}`);
        }
    } catch (error) {
        console.error('获取本地数据失败:', error);
        showAlert('error', '获取本地数据失败');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-sync-alt me-1"></i>刷新数据';
        }
    }
}

/**
 * 更新交易对选择器
 */
function updateSymbolsSelect() {
    allSymbols = [...new Set(localDataRows.map(row => row.symbol))].sort();
    renderSymbolsSelect();
}

function renderSymbolsSelect() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    if (!symbolsSelect) return;

    const selectedSymbols = new Set(
        Array.from(symbolsSelect.selectedOptions).map(opt => opt.value)
    );

    const query = (document.getElementById('symbolSearchInput')?.value || '').trim().toLowerCase();
    const excludeStablecoins = !!document.getElementById('excludeStablecoinsFilter')?.checked;
    const viewSymbols = allSymbols.filter(symbol => {
        if (excludeStablecoins && isStablecoinSymbol(symbol)) return false;
        if (query && !String(symbol).toLowerCase().includes(query)) return false;
        return true;
    });

    symbolsSelect.innerHTML = '';
    viewSymbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol;
        const stableTag = isStablecoinSymbol(symbol) ? ' (稳定币)' : '';
        option.textContent = `${symbol}${stableTag}`;
        if (isStablecoinSymbol(symbol)) {
            option.style.color = '#888';
        }
        if (selectedSymbols.has(symbol)) {
            option.selected = true;
        }
        symbolsSelect.appendChild(option);
    });

    updateSelectedSymbolCount();
    console.log('交易对选择器已更新，选项数:', symbolsSelect.options.length);
}

/**
 * 更新时间框架选择器
 */
function updateTimeframesSelect() {
    const timeframesSelect = document.getElementById('timeframesSelect');
    if (!timeframesSelect) return;
    
    console.log('更新时间框架选择器，数据行数:', localDataRows.length);
    
    // 获取唯一的时间框架
    const timeframes = [...new Set(localDataRows.map(row => row.timeframe))];
    console.log('找到的时间框架:', timeframes);
    
    // 按时间长度排序
    const sortedTimeframes = timeframes.sort((a, b) => {
        const aMinutes = timeframeToMinutes(a);
        const bMinutes = timeframeToMinutes(b);
        return aMinutes - bMinutes;
    });
    
    console.log('排序后的时间框架:', sortedTimeframes);
    
    timeframesSelect.innerHTML = '<option value="">请选择...</option>';
    sortedTimeframes.forEach(timeframe => {
        const option = document.createElement('option');
        option.value = timeframe;
        option.textContent = timeframe;
        timeframesSelect.appendChild(option);
    });
    
    console.log('时间框架选择器已更新，选项数:', timeframesSelect.options.length);
}

/**
 * 时间框架转换为分钟数
 */
function timeframeToMinutes(timeframe) {
    const unit = timeframe.slice(-1);
    const value = parseInt(timeframe.slice(0, -1));
    
    switch (unit) {
        case 'm': return value;
        case 'h': return value * 60;
        case 'd': return value * 1440;
        default: return 0;
    }
}

/**
 * 更新交易对选择后的时间框架（使用包含式逻辑，参考因子评估页面）
 */
function updateTimeframesForSelection() {
    const timeframesSelect = document.getElementById('timeframesSelect');
    const selectedSymbols = Array.from(document.getElementById('symbolsSelect').selectedOptions).map(opt => opt.value);
    
    if (!localDataRows || localDataRows.length === 0) return;
    
    let timeframes = [];
    
    // 包含式（并集）展示：覆盖所选交易对的所有可用时间框架
    if (!selectedSymbols.length) {
        // 未选择交易对时，展示全部时间框架
        timeframes = [...new Set(localDataRows.map(r => r.timeframe))];
    } else {
        const union = new Set();
        selectedSymbols.forEach(sym => {
            localDataRows.filter(r => r.symbol === sym).forEach(r => union.add(r.timeframe));
        });
        timeframes = [...union];
    }
    
    // 按时间框架大小排序（从小到大）
    timeframes.sort((a, b) => timeframeToMinutes(a) - timeframeToMinutes(b));
    
    // 记录原选择，尽量保留
    const prev = new Set(Array.from(timeframesSelect.selectedOptions).map(opt => opt.value));
    
    timeframesSelect.innerHTML = '';
    timeframes.forEach(tf => {
        const option = document.createElement('option');
        option.value = tf;
        option.textContent = tf;
        if (prev.has(tf)) option.selected = true;
        timeframesSelect.appendChild(option);
    });
    
    // 更新时间范围（根据最新选择的交易对与时间框架）
    updateRangeForSelection();
}

/**
 * 根据当前选择（交易对/时间框架）动态更新可用时间范围滑条（参考因子评估页面）
 */
function updateRangeForSelection() {
    if (!localDataRows || localDataRows.length === 0) return;
    
    const selectedSymbols = Array.from(document.getElementById('symbolsSelect').selectedOptions).map(opt => opt.value);
    const selectedTimeframes = Array.from(document.getElementById('timeframesSelect').selectedOptions).map(opt => opt.value);
    
    let rows = localDataRows;
    
    if (selectedSymbols.length) {
        rows = rows.filter(r => selectedSymbols.includes(r.symbol));
    }
    
    if (selectedTimeframes.length) {
        rows = rows.filter(r => selectedTimeframes.includes(r.timeframe));
    }
    
    const starts = rows.map(r => r.date_range && r.date_range.start).filter(Boolean).map(s => new Date(s));
    const ends = rows.map(r => r.date_range && r.date_range.end).filter(Boolean).map(s => new Date(s));
    
    if (starts.length && ends.length) {
        const minDate = new Date(Math.min(...starts));
        const maxDate = new Date(Math.max(...ends));
        setupRangeSlider(minDate, maxDate);
    }
}

/**
 * 设置范围滑块（参考因子评估页面）
 */
function setupRangeSlider(startDate, endDate) {
    const container = document.getElementById('rangeSlider');
    const info = document.getElementById('rangeInfo');
    
    if (container.noUiSlider) {
        container.noUiSlider.destroy();
    }
    
    const totalMs = endDate - startDate;
    
    noUiSlider.create(container, {
        start: [0, 100],
        connect: true,
        step: 1,
        range: { min: 0, max: 100 }
    });
    
    const update = (values) => {
        const sv = parseInt(values[0], 10), ev = parseInt(values[1], 10);
        const s = new Date(startDate.getTime() + (sv/100)*totalMs);
        const e = new Date(startDate.getTime() + (ev/100)*totalMs);
        const startStr = s.toISOString().slice(0, 10);
        const endStr = e.toISOString().slice(0, 10);

        const startDateEl = document.getElementById('startDate');
        const endDateEl = document.getElementById('endDate');
        const startDateDisplayEl = document.getElementById('startDateDisplay');
        const endDateDisplayEl = document.getElementById('endDateDisplay');

        if (startDateEl) startDateEl.value = startStr;
        if (endDateEl) endDateEl.value = endStr;
        if (startDateDisplayEl) startDateDisplayEl.value = startStr;
        if (endDateDisplayEl) endDateDisplayEl.value = endStr;
        updateRangeInfo(`${startStr} ~ ${endStr}`);
    };
    
    container.noUiSlider.on('update', update);
    update([0, 100]);
}

function onManualDateInputChanged() {
    const startDateDisplay = document.getElementById('startDateDisplay');
    const endDateDisplay = document.getElementById('endDateDisplay');
    const startDate = startDateDisplay?.value;
    const endDate = endDateDisplay?.value;
    if (!startDate || !endDate) return;

    if (startDate > endDate) {
        showAlert('warning', '开始日期不能晚于结束日期');
        return;
    }

    const startHidden = document.getElementById('startDate');
    const endHidden = document.getElementById('endDate');
    if (startHidden) startHidden.value = startDate;
    if (endHidden) endHidden.value = endDate;
    updateRangeInfo(`${startDate} ~ ${endDate}`);
}

/**
 * 更新范围信息显示
 */
function updateRangeInfo(message) {
    const rangeInfo = document.getElementById('rangeInfo');
    if (rangeInfo) {
        rangeInfo.textContent = message;
    }
}

function recommendStartDate() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    const selectedSymbols = Array.from(symbolsSelect?.selectedOptions || []).map(opt => opt.value);
    if (selectedSymbols.length === 0) {
        showAlert('warning', '请先选择交易对');
        return;
    }
    if (!localDataRows || localDataRows.length === 0) {
        showAlert('warning', '本地数据未加载，请先刷新数据');
        return;
    }

    const timeframesSelect = document.getElementById('timeframesSelect');
    const selectedTimeframes = Array.from(timeframesSelect?.selectedOptions || []).map(opt => opt.value);
    let rows = localDataRows;
    if (selectedTimeframes.length > 0) {
        rows = rows.filter(r => selectedTimeframes.includes(r.timeframe));
    }

    const symbolStartDates = [];
    const missingSymbols = [];

    selectedSymbols.forEach(sym => {
        const matchingRows = rows.filter(r => r.symbol === sym);
        const row = matchingRows.length > 0 ? matchingRows[0] : null;
        if (row && row.date_range && row.date_range.start) {
            symbolStartDates.push({
                symbol: sym,
                startDate: row.date_range.start,
                endDate: row.date_range.end || '?'
            });
        } else {
            missingSymbols.push(sym);
        }
    });

    if (symbolStartDates.length === 0) {
        showAlert('error', '所选交易对均无本地数据');
        return;
    }

    const allStarts = symbolStartDates.map(s => new Date(s.startDate).getTime());
    const recommendedTs = Math.max(...allStarts);
    const recommendedDate = new Date(recommendedTs);
    const dateStr = recommendedDate.toISOString().slice(0, 10);

    let msg = `推荐开始日期: ${dateStr}\n\n`;
    msg += `已选 ${selectedSymbols.length} 个交易对中，${symbolStartDates.length} 个有本地数据`;
    if (missingSymbols.length > 0) {
        msg += `，${missingSymbols.length} 个无数据: ${missingSymbols.slice(0, 5).join(', ')}${missingSymbols.length > 5 ? ' 等' : ''}`;
    }
    msg += `\n\n该日期可确保所有有数据的交易对均有覆盖`;

    const earliestSymbols = symbolStartDates.filter(s => new Date(s.startDate).getTime() === recommendedTs);
    if (earliestSymbols.length > 0 && earliestSymbols.length <= 5) {
        msg += `\n\n最晚开始的交易对: ${earliestSymbols.map(s => `${s.symbol} (${s.startDate.slice(0,10)})`).join(', ')}`;
    }

    if (confirm(msg + '\n\n是否应用此日期？')) {
        const startDateDisplay = document.getElementById('startDateDisplay');
        const startDateHidden = document.getElementById('startDate');
        if (startDateDisplay) startDateDisplay.value = dateStr;
        if (startDateHidden) startDateHidden.value = dateStr;
        onManualDateInputChanged();
    }
}

async function loadStablecoins() {
    try {
        const response = await fetch('/api/data/stablecoins');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json();
        if (result.success) {
            stablecoins = new Set((result.data || []).map(item => String(item).toUpperCase()));
            return;
        }
    } catch (error) {
        console.warn('加载稳定币列表失败，使用默认列表', error);
    }
    stablecoins = new Set(['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDD', 'FRAX', 'USDP', 'GUSD']);
}

function isStablecoinSymbol(symbol) {
    const text = String(symbol || '').toUpperCase();
    if (!text) return false;

    let base;
    if (text.includes('_')) {
        base = text.split('_')[0];
    } else {
        base = text.replace(/USDT?$/i, '');
    }
    return stablecoins.has(base.toUpperCase());
}

function filterSymbolsByKeyword() {
    renderSymbolsSelect();
}

function filterSymbolsByStablecoin() {
    renderSymbolsSelect();
}

function updateSelectedSymbolCount() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    const countEl = document.getElementById('selectedSymbolCount');
    if (!symbolsSelect || !countEl) return;
    countEl.textContent = String(symbolsSelect.selectedOptions.length);
}

function selectAllSymbols() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    if (!symbolsSelect) return;
    Array.from(symbolsSelect.options).forEach(opt => { opt.selected = true; });
    updateSelectedSymbolCount();
    updateTimeframesForSelection();
}

function selectAllSymbolsExcludeStablecoins() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    if (!symbolsSelect) return;
    Array.from(symbolsSelect.options).forEach(opt => {
        opt.selected = !isStablecoinSymbol(opt.value);
    });
    updateSelectedSymbolCount();
    updateTimeframesForSelection();
}

function clearSymbolSelection() {
    const symbolsSelect = document.getElementById('symbolsSelect');
    if (!symbolsSelect) return;
    Array.from(symbolsSelect.options).forEach(opt => { opt.selected = false; });
    updateSelectedSymbolCount();
    updateTimeframesForSelection();
}

function normalizeSymbolForMatch(rawSymbol) {
    const raw = String(rawSymbol || '').trim().toUpperCase();
    if (!raw) return '';
    return raw.replace(/[\s_\-\/]/g, '');
}

function buildSymbolAliasMap(symbolList) {
    const aliasMap = new Map();
    (symbolList || []).forEach(original => {
        const upper = String(original || '').toUpperCase();
        const normalized = normalizeSymbolForMatch(upper);
        if (upper && !aliasMap.has(upper)) aliasMap.set(upper, original);
        if (normalized && !aliasMap.has(normalized)) aliasMap.set(normalized, original);
    });
    return aliasMap;
}

function importSymbolList() {
    const inputEl = document.getElementById('symbolListInput');
    const resultEl = document.getElementById('importResult');
    if (!inputEl || !resultEl) return;

    const input = inputEl.value.trim();
    if (!input) {
        resultEl.innerHTML = '<span class="text-warning">请输入交易对列表</span>';
        return;
    }

    if (!allSymbols || allSymbols.length === 0) {
        resultEl.innerHTML = '<span class="text-danger">请先加载交易对数据</span>';
        return;
    }

    let symbols = [];
    if (input.startsWith('[') && input.endsWith(']')) {
        try {
            const parsed = JSON.parse(input);
            if (Array.isArray(parsed)) {
                symbols = parsed.map(item => String(item).trim().toUpperCase());
            }
        } catch (error) {
            resultEl.innerHTML = '<span class="text-danger">JSON格式解析失败，请检查格式</span>';
            return;
        }
    } else {
        symbols = input.split(/[,\s\n]+/)
            .map(item => item.trim().toUpperCase())
            .filter(Boolean);
    }

    if (symbols.length === 0) {
        resultEl.innerHTML = '<span class="text-warning">未识别到有效的交易对</span>';
        return;
    }

    const symbolAliasMap = buildSymbolAliasMap(allSymbols);
    const matchedSymbols = [];
    const unmatchedSymbols = [];
    const matchedSet = new Set();

    symbols.forEach(item => {
        const normalized = normalizeSymbolForMatch(item);
        const matched = symbolAliasMap.get(item) || symbolAliasMap.get(normalized);
        if (matched) {
            if (!matchedSet.has(matched)) {
                matchedSet.add(matched);
                matchedSymbols.push(matched);
            }
        } else {
            unmatchedSymbols.push(item);
        }
    });

    const excludeStablecoinsFilter = document.getElementById('excludeStablecoinsFilter');
    if (excludeStablecoinsFilter?.checked) {
        excludeStablecoinsFilter.checked = false;
    }
    renderSymbolsSelect();
    clearSymbolSelection();

    const symbolsSelect = document.getElementById('symbolsSelect');
    if (symbolsSelect) {
        Array.from(symbolsSelect.options).forEach(opt => {
            if (matchedSet.has(opt.value)) {
                opt.selected = true;
            }
        });
    }

    updateSelectedSymbolCount();
    updateTimeframesForSelection();

    let resultHtml = '';
    if (matchedSymbols.length > 0) {
        resultHtml += `<span class="text-success">成功匹配 ${matchedSymbols.length} 个交易对</span>`;
    }
    if (unmatchedSymbols.length > 0) {
        const showCount = Math.min(5, unmatchedSymbols.length);
        const more = unmatchedSymbols.length > showCount ? ` 等${unmatchedSymbols.length}个` : '';
        resultHtml += `<br><span class="text-warning">未匹配: ${unmatchedSymbols.slice(0, showCount).join(', ')}${more}</span>`;
    }
    resultEl.innerHTML = resultHtml;
}

/**
 * 处理挖掘表单提交
 */
async function handleMiningSubmit(event) {
    event.preventDefault();
    console.log('挖掘表单提交, 模式:', currentMiningMode);

    if (currentMiningMode === 'cross_sectional') {
        await handleCrossSectionalMining();
        return;
    }

    if (currentMiningMode === 'cross_sectional_rl') {
        await handleRLCrossSectionalMining();
        return;
    }

    try {
        // 获取表单数据
        const formData = new FormData(event.target);
        const miningData = {
            symbols: getSelectedValues('symbolsSelect'),
            timeframes: getSelectedValues('timeframesSelect'),
            selected_algorithms: getSelectedAlgorithms(),
            start_date: formData.get('startDate'),
            end_date: formData.get('endDate'),
            optimization_method: formData.get('optimizationMethod'),
            max_factors: parseInt(formData.get('maxFactors')) || 15,
            min_ic: parseFloat(formData.get('minIC')) || 0.02,
            min_ir: parseFloat(formData.get('minIR')) || 0.1,
            min_sample_size: parseInt(formData.get('minSampleSize')) || 30
        };
        
        console.log('挖掘配置:', miningData);
        
        // 验证数据
        if (miningData.symbols.length === 0) {
            showAlert('error', '请选择至少一个交易对');
            return;
        }
        
        if (miningData.timeframes.length === 0) {
            showAlert('error', '请选择至少一个时间框架');
            return;
        }
        
        if (miningData.selected_algorithms.length === 0) {
            showAlert('error', '请选择至少一个算法');
            return;
        }
        
        // 更新按钮状态
        updateStartButton(true);
        
        // 显示等待状态
        showWaitingState();
        
        // 启动挖掘
        const response = await fetch('/api/mining/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(miningData)
        });
        
        const result = await response.json();
        console.log('挖掘启动结果:', result);
        
        if (result.success) {
            // 保存会话ID
            miningSession = result.session_id;
            
            // 显示预估时间信息
            if (result.estimated_time) {
                const estimatedTime = formatTime(result.estimated_time);
                showAlert('info', `挖掘已启动！预估总时间: ${estimatedTime}`);
                
                // 更新预估时间显示
                const timeDisplay = document.getElementById('timeDisplay');
                if (timeDisplay) {
                    timeDisplay.textContent = `预估总时间: ${estimatedTime}`;
                }
                
                // 更新系统信息显示
                if (result.system_info) {
                    const systemDisplay = document.getElementById('systemDisplay');
                    if (systemDisplay) {
                        const { cpu_count, memory_gb, memory_percent } = result.system_info;
                        systemDisplay.textContent = `CPU: ${cpu_count}核 | 内存: ${memory_gb}GB (${memory_percent}%)`;
                    }
                }
            }
            
            // 显示进度界面
            showMiningProgress();
            
            // 启动进度监控
            startProgressMonitoring(result.session_id);
            
        } else {
            // 挖掘失败
            showAlert('error', result.error || '挖掘启动失败');
        }
        
    } catch (error) {
        console.error('挖掘失败:', error);
        showAlert('error', `挖掘过程中发生错误: ${error.message}`);
    } finally {
        // 恢复按钮状态
        updateStartButton(false);
    }
}

/**
 * 获取选择框的选中值
 */
function getSelectedValues(selectId) {
    if (selectId === 'factorTypes') {
        // 特殊处理因子类型复选框（已废弃，保留兼容性）
        const checkboxes = document.querySelectorAll('input[name="factorType"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }
    
    const select = document.getElementById(selectId);
    if (!select) return [];
    
    const selectedOptions = Array.from(select.selectedOptions);
    return selectedOptions.map(option => option.value).filter(value => value);
}

/**
 * 验证挖掘表单
 */
function validateMiningForm() {
    const symbols = getSelectedValues('symbolsSelect');
    const timeframes = getSelectedValues('timeframesSelect');
    const selectedAlgorithms = getSelectedAlgorithms();
    
    if (symbols.length === 0) {
        showAlert('error', '请选择至少一个交易对');
        return false;
    }
    
    if (timeframes.length === 0) {
        showAlert('error', '请选择至少一个时间框架');
        return false;
    }
    
    if (selectedAlgorithms.length === 0) {
        showAlert('error', '请选择至少一个算法');
        return false;
    }
    
    return true;
}

/**
 * 显示等待状态
 */
function showWaitingState() {
    const waitingState = document.getElementById('waitingState');
    if (waitingState) {
        waitingState.style.display = 'block';
    }
    
    // 隐藏其他状态
    const miningProgress = document.getElementById('miningProgress');
    if (miningProgress) {
        miningProgress.style.display = 'none';
    }
    
    const miningResults = document.getElementById('miningResults');
    if (miningResults) {
        miningResults.style.display = 'none';
    }
}

/**
 * 获取挖掘表单数据
 */
function getMiningFormData() {
    const exchange = document.getElementById('exchangeSelect').value;
    const tradeType = document.getElementById('tradeTypeSelect').value;
    const symbols = Array.from(document.getElementById('symbolsSelect').selectedOptions).map(opt => opt.value);
    const timeframes = Array.from(document.getElementById('timeframesSelect').selectedOptions).map(opt => opt.value);
    const maxFactors = parseInt(document.getElementById('maxFactors').value) || 15;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    return {
        exchange: exchange,
        trade_type: tradeType,
        symbols: symbols,
        timeframes: timeframes,
        max_factors: maxFactors,
        start_date: startDate,
        end_date: endDate,
        optimization_method: 'greedy'
    };
}


/**
 * 开始因子挖掘
 */
async function startMining(formData) {
    try {
        console.log('开始因子挖掘...');
        
        // 显示进度界面
        showMiningProgress();
        
        // 更新按钮状态
        updateStartButton(true);
        
        // 调用挖掘API
        const response = await fetch('/api/mining/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('挖掘API响应:', result);
        
        if (result.success) {
            // 挖掘成功，保存会话ID
            miningSession = result.session_id;
            showAlert('success', '因子挖掘已启动，正在监控进度...');
            console.log('挖掘会话ID:', result.session_id);
            
            // 添加调试信息
            addDebugInfo('挖掘任务已启动', 'success');
            addDebugInfo(`会话ID: ${result.session_id}`, 'info');
            
            // 重置进度步骤
            resetProgressSteps();
            
            // 启动进度监控
            startProgressMonitoring(result.session_id);
        } else {
            // 挖掘失败
            addDebugInfo(`挖掘启动失败: ${result.error || '未知错误'}`, 'error');
            showAlert('error', result.error || '挖掘启动失败');
        }
        
    } catch (error) {
        console.error('挖掘失败:', error);
        showAlert('error', `挖掘过程中发生错误: ${error.message}`);
    } finally {
        // 恢复按钮状态
        updateStartButton(false);
    }
}

/**
 * 显示挖掘进度界面
 */
function showMiningProgress() {
    const waitingState = document.getElementById('waitingState');
    if (waitingState) {
        waitingState.style.display = 'none';
    }
    
    const miningProgress = document.getElementById('miningProgress');
    if (miningProgress) {
        miningProgress.style.display = 'block';
    }

    const stopBtn = document.getElementById('stopMiningBtn');
    if (stopBtn) {
        stopBtn.style.display = 'inline-block';
    }
    
    resetSubProgress();
}

/**
 * 重置进度步骤
 */
function resetProgressSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
    console.log('重置进度步骤:', steps);
    
    steps.forEach(stepId => {
        const stepElement = document.getElementById(stepId);
        if (stepElement) {
            const progressBar = stepElement.querySelector('.progress-fill');
            const stepIcon = stepElement.querySelector('.step-icon');
            const stepDetails = stepElement.querySelector('.step-details');
            
            if (progressBar) {
                progressBar.style.width = '0%';
                console.log(`重置进度条 ${stepId}: 0%`);
            }
            if (stepIcon) {
                stepIcon.className = 'step-icon pending';
                stepIcon.innerHTML = '<i class="fas fa-clock"></i>';
                console.log(`重置图标 ${stepId}: 等待状态`);
            }
            if (stepDetails) {
                stepDetails.textContent = '等待开始...';
                console.log(`重置详情 ${stepId}: 等待开始...`);
            }
        } else {
            console.error(`找不到步骤元素: ${stepId}`);
        }
    });
    
    // 重置总体进度
    updateOverallProgress(0, null, [], null, null);
}

/**
 * 启动进度监控
 */
function startProgressMonitoring(sessionId) {
    console.log('启动进度监控，会话ID:', sessionId);
    
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    if (progressEventSource) {
        progressEventSource.close();
    }
    
    // 使用Server-Sent Events进行实时进度更新
    progressEventSource = new EventSource(`/api/mining/progress/${sessionId}`);
    
    progressEventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('收到进度更新:', data);
            
            if (data.success !== false) {
                updateProgressDisplay(data);
                
                if (data.status === 'completed' || data.status === 'stopped') {
                    handleMiningCompleted(sessionId);
                } else if (data.status === 'error' || data.status === 'failed') {
                    handleMiningError(data.error || data.message || '挖掘任务失败');
                }
            } else {
                addDebugInfo(`进度更新失败: ${data.error || '未知错误'}`, 'error');
            }
        } catch (error) {
            console.error('解析进度数据失败:', error);
            addDebugInfo(`解析进度数据失败: ${error.message}`, 'error');
        }
    };
    
    progressEventSource.onerror = function(error) {
        console.error('进度流错误:', error);
        addDebugInfo('SSE连接失败，切换到轮询模式', 'warning');
        // 如果SSE失败，回退到轮询
        fallbackToPolling(sessionId);
    };
}

/**
 * 回退到轮询方式
 */
function fallbackToPolling(sessionId) {
    console.log('回退到轮询方式');
    addDebugInfo('开始轮询模式监控进度', 'info');
    
    if (progressEventSource) {
        progressEventSource.close();
        progressEventSource = null;
    }
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/mining/status/${sessionId}`);
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    updateProgressDisplay(data);
                    
                    if (data.status === 'completed' || data.status === 'stopped') {
                        handleMiningCompleted(sessionId);
                    } else if (data.status === 'error' || data.status === 'failed') {
                        handleMiningError(data.error || data.message || '挖掘任务失败');
                    }
                } else {
                    addDebugInfo(`轮询获取状态失败: ${data.error || '未知错误'}`, 'error');
                }
            } else {
                addDebugInfo(`轮询HTTP错误: ${response.status}`, 'error');
            }
        } catch (error) {
            console.error('获取进度失败:', error);
            addDebugInfo(`轮询获取进度失败: ${error.message}`, 'error');
        }
    }, 1000); // 每秒更新一次
}

/**
 * 更新进度显示（简化版）
 */
function updateProgressDisplay(data) {
    const { progress, current_step, messages, message } = data;
    console.log('更新进度显示:', { progress, current_step, messages, message });
    
    const overallProgress = progress || 0;
    updateOverallProgress(overallProgress, current_step, messages);
    
    addDebugInfo(`[${new Date().toLocaleTimeString()}] 进度更新: ${overallProgress}% - ${current_step || '未知步骤'}`);

    if (message) {
        addDebugInfo(`[${new Date().toLocaleTimeString()}] ${message}`);
    }

    if (messages && messages.length > 0) {
        messages.forEach(msg => {
            addDebugInfo(`[${new Date().toLocaleTimeString()}] ${msg.message || msg}`);
        });
    }
}

/**
 * 更新总体进度（简化版）
 */
function updateOverallProgress(progress, currentStep, messages) {
    const overallProgressBar = document.getElementById('overallProgress');
    const progressPercent = document.getElementById('progressPercent');
    const progressBadge = document.getElementById('progressBadge');
    
    if (overallProgressBar) {
        overallProgressBar.style.width = `${progress}%`;
    }
    
    if (progressPercent) {
        progressPercent.textContent = `${progress}%`;
    }
    
    if (progressBadge) {
        if (progress === 0) {
            progressBadge.textContent = '准备中';
            progressBadge.className = 'badge bg-secondary ms-2';
        } else if (progress === 100) {
            progressBadge.textContent = '完成';
            progressBadge.className = 'badge bg-success ms-2';
        } else {
            progressBadge.textContent = '运行中';
            progressBadge.className = 'badge bg-primary ms-2';
        }
    }
}

/**
 * 添加调试信息
 */
function addDebugInfo(message, type = 'info') {
    const debugInfo = document.getElementById('debugInfo');
    if (!debugInfo) return;
    
    const debugItem = document.createElement('div');
    debugItem.className = `debug-item ${type}`;
    
    const time = new Date().toLocaleTimeString();
    debugItem.innerHTML = `
        <span class="debug-time">[${time}]</span>
        <span class="debug-message">${message}</span>
    `;
    
    debugInfo.appendChild(debugItem);
    
    // 限制调试信息数量，避免过多
    const items = debugInfo.querySelectorAll('.debug-item');
    if (items.length > 50) {
        items[0].remove();
    }
    
    // 自动滚动到底部
    debugInfo.scrollTop = debugInfo.scrollHeight;
}

/**
 * 清空调试信息
 */
function clearDebugInfo() {
    const debugInfo = document.getElementById('debugInfo');
    if (debugInfo) {
        debugInfo.innerHTML = `
            <div class="debug-item">
                <span class="debug-time">[${new Date().toLocaleTimeString()}]</span>
                <span class="debug-message">调试信息已清空</span>
            </div>
        `;
    }
}

/**
 * 格式化时间显示
 */
function formatTime(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)}秒`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.round(seconds % 60);
        return `${minutes}分${remainingSeconds}秒`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const remainingMinutes = Math.floor((seconds % 3600) / 60);
        return `${hours}小时${remainingMinutes}分`;
    }
}

/**
 * 处理挖掘完成
 */
async function handleMiningCompleted(sessionId) {
    console.log('挖掘完成，会话ID:', sessionId);
    
    // 添加调试信息
    addDebugInfo('挖掘任务完成！', 'success');
    
    // 停止进度监控
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    if (progressEventSource) {
        progressEventSource.close();
        progressEventSource = null;
    }
    
    // 更新按钮状态
    updateStartButton(false);
    
    // 显示完成提示
    showAlert('success', '因子挖掘完成！');
    
    try {
        // 获取完整的挖掘结果
        addDebugInfo('正在获取挖掘结果...', 'info');
        console.log('获取完整挖掘结果...');
        const response = await fetch(`/api/mining/result/${sessionId}`);
        if (response.ok) {
            // 获取响应文本
            const responseText = await response.text();
            console.log('挖掘结果响应文本:', responseText);
            
            // 清理NaN值，替换为null
            const cleanedText = responseText.replace(/: NaN/g, ': null');
            console.log('清理后的响应文本:', cleanedText);
            
            let resultData;
            try {
                // 尝试解析清理后的JSON
                resultData = JSON.parse(cleanedText);
            } catch (parseError) {
                console.error('JSON解析失败:', parseError);
                console.error('清理后的响应文本:', cleanedText);
                addDebugInfo(`JSON解析失败: ${parseError.message}`, 'error');
                throw new Error(`JSON解析失败: ${parseError.message}`);
            }
            
            console.log('挖掘结果数据:', resultData);
            
            if (resultData.success !== false) {
                // 显示结果
                addDebugInfo('挖掘结果获取成功，正在显示...', 'success');

                const isCrossSectional = resultData.results?.mode === 'cross_sectional'
                    || resultData.config?.mode === 'cross_sectional';
                const isRL = resultData.results?.mode === 'cross_sectional_rl'
                    || resultData.config?.mode === 'cross_sectional_rl';

                if (isRL) {
                    showMiningResults(resultData);
                } else if (isCrossSectional) {
                    showMiningResults(resultData);
                } else {
                    showMiningResults(resultData);
                }

                // 追加加载对比报告
                loadDiffReport(sessionId);
            } else {
                console.error('获取挖掘结果失败:', resultData.error);
                addDebugInfo(`获取挖掘结果失败: ${resultData.error}`, 'error');
                showAlert('error', `获取挖掘结果失败: ${resultData.error}`);
            }
        } else {
            addDebugInfo(`HTTP错误: ${response.status}`, 'error');
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    } catch (error) {
        console.error('获取挖掘结果失败:', error);
        addDebugInfo(`获取挖掘结果失败: ${error.message}`, 'error');
        showAlert('error', `获取挖掘结果失败: ${error.message}`);
    }
    
    // 刷新挖掘历史
    loadMiningHistory();
}

/**
 * 处理挖掘错误
 */
function handleMiningError(error) {
    console.error('挖掘错误:', error);
    
    // 添加调试信息
    addDebugInfo(`挖掘任务失败: ${error}`, 'error');
    
    // 停止进度监控
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    
    if (progressEventSource) {
        progressEventSource.close();
        progressEventSource = null;
    }
    
    // 更新按钮状态
    updateStartButton(false);
    
    // 显示错误提示
    showAlert('error', `挖掘失败: ${error}`);
}

async function stopMining() {
    if (!miningSession) {
        showAlert('warning', '没有正在运行的挖掘任务');
        return;
    }

    const stopBtn = document.getElementById('stopMiningBtn');
    if (stopBtn) {
        stopBtn.disabled = true;
        stopBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>停止中...';
    }

    try {
        const sessionId = typeof miningSession === 'string' ? miningSession : miningSession.session_id;
        const isStandard = currentMiningMode === 'standard';
        const endpoint = isStandard
            ? `/api/mining/stop/${sessionId}`
            : `/api/mining/cross_sectional/stop/${sessionId}`;

        const response = await fetch(endpoint, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            addDebugInfo('停止信号已发送，等待任务中止...', 'warning');
            showAlert('warning', '正在停止挖掘任务...');
        } else {
            addDebugInfo(`停止失败: ${result.error || '未知错误'}`, 'error');
            showAlert('error', result.error || '停止挖掘失败');
            if (stopBtn) {
                stopBtn.disabled = false;
                stopBtn.innerHTML = '<i class="fas fa-stop me-1"></i>停止挖掘';
            }
        }
    } catch (error) {
        console.error('停止挖掘失败:', error);
        addDebugInfo(`停止挖掘失败: ${error.message}`, 'error');
        showAlert('error', `停止挖掘失败: ${error.message}`);
        if (stopBtn) {
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<i class="fas fa-stop me-1"></i>停止挖掘';
        }
    }
}

/**
 * 显示挖掘结果
 */
function showMiningResults(data) {
    const miningProgress = document.getElementById('miningProgress');
    if (miningProgress) {
        miningProgress.style.display = 'none';
    }

    const miningResults = document.getElementById('miningResults');
    if (miningResults) {
        miningResults.style.display = 'block';

        const isCrossSectional = data.results?.mode === 'cross_sectional'
            || data.config?.mode === 'cross_sectional';
        const isRL = data.results?.mode === 'cross_sectional_rl'
            || data.config?.mode === 'cross_sectional_rl';

        if (isRL) {
            showRLResults(data);
        } else if (isCrossSectional) {
            showCrossSectionalResults(data);
        } else {
            updateResultsOverview(data);
            updateResultsTable(data);
        }

        ensureDiffContainer();
    }
}

/**
 * 确保对比报告容器存在
 */
function ensureDiffContainer() {
    const miningResults = document.getElementById('miningResults');
    if (!miningResults) return;
    let diffSection = document.getElementById('diffSection');
    if (!diffSection) {
        diffSection = document.createElement('div');
        diffSection.id = 'diffSection';
        diffSection.className = 'mt-4';
        diffSection.innerHTML = `
            <h5>因子对比报告</h5>
            <div id="diffSummary" class="mb-2 small text-muted"></div>
            <div id="diffTableWrap" class="table-responsive"></div>
            <div class="mt-2">
                <button id="saveSelectedBtn" class="btn btn-primary btn-sm" disabled>保存选中因子到因子库</button>
                <span id="saveSelectedHint" class="ms-2 text-muted small"></span>
            </div>
        `;
        miningResults.appendChild(diffSection);
        const btn = diffSection.querySelector('#saveSelectedBtn');
        if (btn) {
            btn.addEventListener('click', onSaveSelectedFactors);
        }
    }
}

/**
 * 加载对比报告
 */
async function loadDiffReport(sessionId) {
    try {
        const res = await fetch(`/api/mining/diff/${sessionId}`);
        if (!res.ok) {
            console.warn('diff接口响应非200', res.status);
            return;
        }
        const payload = await res.json();
        if (payload && payload.success) {
            console.log('diff payload:', payload);
            renderDiffReport(sessionId, payload.diff_report || {});
        }
    } catch (e) {
        console.warn('加载对比报告失败', e);
    }
}

/**
 * 渲染对比报告
 */
function renderDiffReport(sessionId, diff) {
    ensureDiffContainer();
    const summaryEl = document.getElementById('diffSummary');
    const tableWrap = document.getElementById('diffTableWrap');
    const saveBtn = document.getElementById('saveSelectedBtn');
    if (!summaryEl || !tableWrap) return;

    const summary = diff.summary || { total_mined: 0, new: 0, identical: 0, different: 0, missing_artifact: 0 };
    summaryEl.textContent = `总计: ${summary.total_mined}，新增: ${summary.new}，差异: ${summary.different}，相同: ${summary.identical}，缺少模型: ${summary.missing_artifact}`;

    const items = diff.items || [];
    if (!items.length) {
        tableWrap.innerHTML = '<div class="alert alert-secondary">暂无对比数据</div>';
        if (saveBtn) saveBtn.disabled = true;
        return;
    }

    const table = document.createElement('table');
    table.className = 'table table-sm table-striped align-middle';
    table.innerHTML = `
        <thead>
            <tr>
                <th style="width: 32px;"><input type="checkbox" id="diffSelectAll"></th>
                <th>因子ID</th>
                <th>状态</th>
                <th>现有模型签名/函数签名</th>
                <th>新模型签名</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;
    const tbody = table.querySelector('tbody');

    items.forEach(it => {
        const tr = document.createElement('tr');
        const existingSig = (it.existing && (it.existing.model_meta && it.existing.model_meta.signature)) || it.existing.function_signature || '-';
        const newSig = (it.new && it.new.model_meta && it.new.model_meta.signature) || '-';
        const selectable = (it.status === 'new' || it.status === 'different');
        tr.innerHTML = `
            <td><input type="checkbox" class="diffRowChk" data-factor-id="${it.factor_id}" ${selectable ? '' : 'disabled'}></td>
            <td>${it.factor_id}</td>
            <td><span class="badge ${badgeClassForStatus(it.status)}">${labelForStatus(it.status)}</span></td>
            <td><code>${escapeHtml(String(existingSig))}</code></td>
            <td><code>${escapeHtml(String(newSig))}</code></td>
        `;
        tbody.appendChild(tr);
    });

    tableWrap.innerHTML = '';
    tableWrap.appendChild(table);

    // 绑定全选与勾选变化
    const selectAll = table.querySelector('#diffSelectAll');
    const rowChks = table.querySelectorAll('.diffRowChk');
    const updateBtn = () => {
        const anySelected = Array.from(rowChks).some(chk => chk.checked && !chk.disabled);
        if (saveBtn) {
            saveBtn.disabled = !anySelected;
            saveBtn.dataset.sessionId = sessionId;
        }
        const hint = document.getElementById('saveSelectedHint');
        if (hint) {
            const selCount = Array.from(rowChks).filter(chk => chk.checked && !chk.disabled).length;
            hint.textContent = selCount ? `已选择 ${selCount} 个因子` : '';
        }
    };
    if (selectAll) {
        selectAll.addEventListener('change', () => {
            rowChks.forEach(chk => { if (!chk.disabled) chk.checked = selectAll.checked; });
            updateBtn();
        });
    }
    rowChks.forEach(chk => chk.addEventListener('change', updateBtn));
    updateBtn();
}

function badgeClassForStatus(status) {
    switch (status) {
        case 'new': return 'bg-success';
        case 'different': return 'bg-warning text-dark';
        case 'identical': return 'bg-secondary';
        case 'missing_artifact': return 'bg-danger';
        default: return 'bg-secondary';
    }
}

function labelForStatus(status) {
    switch (status) {
        case 'new': return '新增';
        case 'different': return '差异';
        case 'identical': return '相同';
        case 'missing_artifact': return '缺少模型';
        default: return status;
    }
}

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

async function onSaveSelectedFactors() {
    const btn = document.getElementById('saveSelectedBtn');
    if (!btn || btn.disabled) return;
    const sessionId = btn.dataset.sessionId;
    const wrap = document.getElementById('diffTableWrap');
    if (!wrap) return;
    const chks = wrap.querySelectorAll('.diffRowChk');
    const selected = Array.from(chks).filter(chk => chk.checked && !chk.disabled).map(chk => chk.dataset.factorId);
    if (!selected.length) return;
    try {
        btn.disabled = true;
        const resp = await fetch('/api/mining/save_selected_factors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, factor_ids: selected })
        });
        const json = await resp.json();
        if (json && json.success) {
            showAlert('success', `已保存 ${json.saved_count} 个因子定义`);
        } else {
            showAlert('error', `保存失败: ${json && json.message ? json.message : '未知错误'}`);
        }
    } catch (e) {
        showAlert('error', `保存失败: ${e.message}`);
    } finally {
        btn.disabled = false;
    }
}

function renderMinedSaveActions(sessionId, tableContainer) {
    if (!tableContainer || !tableContainer.parentElement) return;
    let actionWrap = document.getElementById('minedSaveActions');
    if (!actionWrap) {
        actionWrap = document.createElement('div');
        actionWrap.id = 'minedSaveActions';
        actionWrap.className = 'mt-2 d-flex align-items-center gap-2';
        actionWrap.innerHTML = `
            <button id="saveMinedSelectedBtn" class="btn btn-success btn-sm" disabled>加入因子库（挖掘因子）</button>
            <span id="saveMinedSelectedHint" class="text-muted small"></span>
        `;
        tableContainer.parentElement.appendChild(actionWrap);
        const btn = document.getElementById('saveMinedSelectedBtn');
        if (btn) btn.addEventListener('click', onSaveMinedSelectedFactors);
    }
    const btn = document.getElementById('saveMinedSelectedBtn');
    if (btn) btn.dataset.sessionId = sessionId || '';
    bindMinedSelectionEvents();
}

function bindMinedSelectionEvents() {
    const btn = document.getElementById('saveMinedSelectedBtn');
    const hint = document.getElementById('saveMinedSelectedHint');
    const selectAll = document.getElementById('minedSelectAll');
    const rowChks = document.querySelectorAll('.minedRowChk');
    if (!btn) return;
    const updateState = () => {
        const checked = Array.from(rowChks).filter(chk => chk.checked).length;
        btn.disabled = checked === 0;
        if (hint) hint.textContent = checked > 0 ? `已选择 ${checked} 个因子` : '';
    };
    if (selectAll) {
        selectAll.addEventListener('change', () => {
            rowChks.forEach(chk => { chk.checked = selectAll.checked; });
            updateState();
        });
    }
    rowChks.forEach(chk => chk.addEventListener('change', updateState));
    updateState();
}

async function onSaveMinedSelectedFactors() {
    const btn = document.getElementById('saveMinedSelectedBtn');
    if (!btn || btn.disabled) return;
    const sessionId = btn.dataset.sessionId;
    const selected = Array.from(document.querySelectorAll('.minedRowChk'))
        .filter(chk => chk.checked)
        .map(chk => chk.dataset.factorId);
    if (!sessionId || !selected.length) return;
    try {
        btn.disabled = true;
        const origText = btn.textContent;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>保存中...';
        const resp = await fetch('/api/mining/save_selected_factors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, factor_ids: selected })
        });
        const json = await resp.json();
        if (json && json.success) {
            showAlert('success', `已加入因子库 ${json.saved_count} 个因子，可在截面评估直接使用`);
            document.querySelectorAll('.minedRowChk').forEach(chk => {
                if (selected.includes(chk.dataset.factorId)) {
                    chk.checked = false;
                    chk.closest('tr')?.classList.add('table-success');
                    setTimeout(() => chk.closest('tr')?.classList.remove('table-success'), 3000);
                }
            });
            const selectAll = document.getElementById('minedSelectAll');
            if (selectAll) selectAll.checked = false;
        } else {
            showAlert('error', `加入因子库失败: ${json && (json.message || json.error) ? (json.message || json.error) : '未知错误'}`);
        }
    } catch (e) {
        showAlert('error', `加入因子库失败: ${e.message}`);
    } finally {
        btn.innerHTML = '加入因子库（挖掘因子）';
        bindMinedSelectionEvents();
    }
}

/**
 * 更新结果概览
 */
function updateResultsOverview(data) {
    console.log('更新结果概览:', data);
    
    try {
        const results = data.results || data;

        const totalFactorsElement = document.getElementById('totalFactors');
        if (totalFactorsElement) {
            totalFactorsElement.textContent = results.total_factors || data.factor_count || data.factors_info?.total_factors || 0;
        }
        
        const selectedFactorsElement = document.getElementById('selectedFactors');
        if (selectedFactorsElement) {
            const opt = results.optimization || data.optimization;
            if (opt && opt.selected_factors) {
                selectedFactorsElement.textContent = opt.selected_factors.length || 0;
            } else {
                selectedFactorsElement.textContent = results.total_factors || 0;
            }
        }
        
        const avgICElement = document.getElementById('avgIC');
        if (avgICElement) {
            const evaluationData = results.evaluation || data.evaluation?.evaluation || data.evaluation;
            if (evaluationData) {
                const factors = Object.values(evaluationData);
                if (factors.length > 0) {
                    const avgIC = factors.reduce((sum, factor) => {
                        const ic = factor.ic_pearson || factor.ic_spearman || 0;
                        return sum + ic;
                    }, 0) / factors.length;
                    avgICElement.textContent = avgIC.toFixed(4);
                }
            }
        }
        
        const avgReturnElement = document.getElementById('avgReturn');
        if (avgReturnElement) {
            const evaluationData = results.evaluation || data.evaluation?.evaluation || data.evaluation;
            if (evaluationData) {
                const factors = Object.values(evaluationData);
                if (factors.length > 0) {
                    const avgReturn = factors.reduce((sum, factor) => {
                        return sum + (factor.long_short_return || 0);
                    }, 0) / factors.length;
                    avgReturnElement.textContent = `${(avgReturn * 100).toFixed(2)}%`;
                }
            }
        }
        
        console.log('结果概览更新完成');
    } catch (error) {
        console.error('更新结果概览失败:', error);
    }
}

/**
 * 更新结果表格
 */
function updateResultsTable(data) {
    console.log('更新结果表格:', data);
    
    try {
        const resultsTableContainer = document.querySelector('#miningResults .table-responsive');
        if (!resultsTableContainer) {
            console.error('找不到结果表格容器');
            return;
        }
        
        resultsTableContainer.innerHTML = '';
        
        const results = data.results || data;
        const evaluationData = results.evaluation || data.evaluation?.evaluation || data.evaluation;
        if (!evaluationData || Object.keys(evaluationData).length === 0) {
            resultsTableContainer.innerHTML = '<div class="alert alert-warning">暂无挖掘结果</div>';
            return;
        }
        
        const table = document.createElement('table');
        table.className = 'table table-sm table-striped table-hover align-middle';
        
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th style="width:32px;"><input type="checkbox" id="minedSelectAll"></th>
                <th>因子名称</th>
                <th>多空收益</th>
                <th>IC (Pearson)</th>
                <th>IC (Spearman)</th>
                <th>胜率</th>
                <th>夏普比率</th>
                <th>类型</th>
            </tr>
        `;
        table.appendChild(thead);
        
        const tbody = document.createElement('tbody');
        
        let algorithmType = '未知';
        const config = data.config || results.config || {};
        if (config.selected_algorithms && config.selected_algorithms.length > 0) {
            algorithmType = config.selected_algorithms[0];
        } else if (results.algorithms_used && results.algorithms_used.length > 0) {
            algorithmType = results.algorithms_used[0];
        }
        
        const sortedFactors = Object.entries(evaluationData).sort(([,a], [,b]) => {
            const returnA = a.long_short_return || 0;
            const returnB = b.long_short_return || 0;
            return returnB - returnA;
        });
        
        sortedFactors.forEach(([factorName, factorData]) => {
            const row = document.createElement('tr');
            const longShortReturn = (factorData.long_short_return || 0) * 100;
            const returnClass = longShortReturn > 0 ? 'text-success' : longShortReturn < 0 ? 'text-danger' : '';
            row.innerHTML = `
                <td><input type="checkbox" class="minedRowChk" data-factor-id="${factorName}"></td>
                <td>${factorName}</td>
                <td class="${returnClass}"><strong>${longShortReturn.toFixed(2)}%</strong></td>
                <td>${(factorData.ic_pearson || 0).toFixed(4)}</td>
                <td>${(factorData.ic_spearman || 0).toFixed(4)}</td>
                <td>${((factorData.win_rate || 0) * 100).toFixed(1)}%</td>
                <td>${(factorData.sharpe_ratio || 0).toFixed(2)}</td>
                <td><span class="badge bg-secondary">${algorithmType}</span></td>
            `;
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        
        resultsTableContainer.appendChild(table);

        const sessionId = typeof miningSession === 'string' ? miningSession : (miningSession?.session_id || data.session_id || '');
        renderMinedSaveActions(sessionId, resultsTableContainer);

        updateCharts(data);
        
        console.log('结果表格更新完成');
    } catch (error) {
        console.error('更新结果表格失败:', error);
    }
}

/**
 * 更新图表
 */
function updateCharts(data) {
    try {
        console.log('开始更新图表:', data);
        
        // 更新类型分布图
        updateTypeChart(data);
        
        // 更新性能分布图
        updatePerformanceChart(data);
        
        console.log('图表更新完成');
    } catch (error) {
        console.error('更新图表失败:', error);
    }
}

/**
 * 更新类型分布图
 */
function updateTypeChart(data) {
    try {
        const canvas = document.getElementById('typeChart');
        if (!canvas) {
            console.error('找不到类型分布图容器');
            return;
        }
        
        const results = data.results || data;
        const config = data.config || results.config || {};

        let algorithmType = '未知';
        if (config.selected_algorithms && config.selected_algorithms.length > 0) {
            algorithmType = config.selected_algorithms[0];
        } else if (results.algorithms_used && results.algorithms_used.length > 0) {
            algorithmType = results.algorithms_used[0];
        }
        
        const evaluationData = results.evaluation || data.evaluation || {};
        const typeCounts = {};
        typeCounts[algorithmType] = Object.keys(evaluationData).length;
        
        // 销毁旧图表
        if (canvas._chartInstance) {
            canvas._chartInstance.destroy();
        }
        
        // 创建新图表
        const ctx = canvas.getContext('2d');
        canvas._chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(typeCounts),
                datasets: [{
                    data: Object.values(typeCounts),
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: '因子类型分布',
                        font: { size: 14 }
                    }
                }
            }
        });
        
        console.log('类型分布图更新完成');
    } catch (error) {
        console.error('更新类型分布图失败:', error);
    }
}

/**
 * 更新性能分布图
 */
function updatePerformanceChart(data) {
    try {
        const canvas = document.getElementById('performanceChart');
        if (!canvas) {
            console.error('找不到性能分布图容器');
            return;
        }
        
        const results = data.results || data;
        const evaluationData = results.evaluation || data.evaluation || {};
        if (!evaluationData || Object.keys(evaluationData).length === 0) {
            return;
        }
        
        // 提取IC值用于性能分布
        const icValues = [];
        Object.values(evaluationData).forEach(factorData => {
            if (factorData.ic_pearson !== undefined) {
                icValues.push(Math.abs(factorData.ic_pearson));
            }
        });
        
        if (icValues.length === 0) {
            return;
        }
        
        // 计算性能分布区间
        const maxIC = Math.max(...icValues);
        const minIC = Math.min(...icValues);
        const range = maxIC - minIC;
        const binCount = 5;
        const binSize = range / binCount;
        
        const bins = new Array(binCount).fill(0);
        const binLabels = [];
        
        for (let i = 0; i < binCount; i++) {
            const start = minIC + i * binSize;
            const end = minIC + (i + 1) * binSize;
            binLabels.push(`${start.toFixed(4)}-${end.toFixed(4)}`);
            
            icValues.forEach(ic => {
                if (ic >= start && ic < end) {
                    bins[i]++;
                }
            });
        }
        
        // 处理最后一个区间（包含最大值）
        bins[binCount - 1] += icValues.filter(ic => ic === maxIC).length;
        
        // 销毁旧图表
        if (canvas._chartInstance) {
            canvas._chartInstance.destroy();
        }
        
        // 创建新图表
        const ctx = canvas.getContext('2d');
        canvas._chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: '因子数量',
                    data: bins,
                    backgroundColor: 'rgba(75, 192, 192, 0.8)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'IC值分布',
                        font: { size: 14 }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '因子数量'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'IC值范围'
                        }
                    }
                }
            }
        });
        
        console.log('性能分布图更新完成');
    } catch (error) {
        console.error('更新性能分布图失败:', error);
    }
}

/**
 * 更新开始按钮状态
 */
function updateStartButton(disabled) {
    const startBtn = document.getElementById('startMiningBtn');
    if (startBtn) {
        startBtn.disabled = disabled;
        startBtn.innerHTML = disabled ? 
            '<i class="fas fa-spinner fa-spin me-2"></i>挖掘中...' : 
            '<i class="fas fa-rocket me-2"></i>开始因子挖掘';
    }

    const stopBtn = document.getElementById('stopMiningBtn');
    if (stopBtn) {
        stopBtn.style.display = disabled ? 'inline-block' : 'none';
    }
}

/**
 * 加载挖掘历史
 */
async function loadMiningHistory() {
    try {
        console.log('开始加载挖掘历史...');
        
        // 检查网络连接
        if (!navigator.onLine) {
            console.error('网络连接不可用');
            return;
        }
        
        const response = await fetch('/api/mining/history');
        console.log('API响应:', response);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // 获取响应文本
        const responseText = await response.text();
        console.log('API响应文本:', responseText);
        
        // 清理NaN值，替换为null
        const cleanedText = responseText.replace(/: NaN/g, ': null');
        console.log('清理后的响应文本:', cleanedText);
        
        let data;
        try {
            // 尝试解析清理后的JSON
            data = JSON.parse(cleanedText);
        } catch (parseError) {
            console.error('JSON解析失败:', parseError);
            console.error('清理后的响应文本:', cleanedText);
            throw new Error(`JSON解析失败: ${parseError.message}`);
        }
        
        console.log('挖掘历史数据:', data);

        if (data.success) {
            console.log(`成功获取历史数据，共 ${data.history?.length || 0} 个会话`);
            updateHistoryTable(data.history);
        } else {
            console.error('加载挖掘历史失败:', data.error);
            // 显示错误信息
            showAlert('error', `加载挖掘历史失败: ${data.error}`);
        }
    } catch (error) {
        console.error('加载挖掘历史失败:', error);
        // 显示错误信息
        showAlert('error', `加载挖掘历史失败: ${error.message}`);
        
        // 尝试显示空状态
        const historyTableBody = document.getElementById('historyTableBody');
        if (historyTableBody) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-muted">
                        <i class="fas fa-exclamation-triangle me-2"></i>加载失败: ${error.message}
                    </td>
                </tr>
            `;
        }
    }
}

/**
 * 更新历史表格
 */
function updateHistoryTable(sessions) {
    console.log('开始更新历史表格，数据:', sessions);
    
    const historyTableBody = document.getElementById('historyTableBody');
    if (!historyTableBody) {
        console.error('找不到历史表格元素');
        showAlert('error', '找不到历史表格元素');
        return;
    }
    
    console.log('找到历史表格元素，开始更新...');
    
    // 根据当前页面预设模式过滤历史记录
    if (window.__MINING_PAGE_PRESET__ && window.__MINING_PAGE_PRESET__.mode && sessions && sessions.length > 0) {
        const presetMode = window.__MINING_PAGE_PRESET__.mode;
        sessions = sessions.filter(session => {
            const modeRaw = session.mode || session.config?.mode || session.results?.mode || 'unknown';
            const safeMode = String(modeRaw).replace(/[<>]/g, '');
            if (presetMode === 'standard') {
                return safeMode === 'standard';
            } else if (presetMode === 'cross_sectional') {
                return safeMode === 'cross_sectional' || safeMode === 'cross_sectional_rl' || safeMode === 'cross_sectional_gp';
            }
            return true;
        });
    }
    
    // 清空现有内容
    historyTableBody.innerHTML = '';
    
    if (!sessions || sessions.length === 0) {
        console.log('没有历史数据，显示空状态');
        historyTableBody.innerHTML = `
            <tr>
                <td colspan="8" class="text-center text-muted">
                    <i class="fas fa-info-circle me-2"></i>暂无挖掘历史
                </td>
            </tr>
        `;
        return;
    }
    
    console.log(`开始处理 ${sessions.length} 个会话...`);
    
    // 添加历史记录
    sessions.forEach((session, index) => {
        try {
            console.log(`处理第 ${index + 1} 个会话:`, session);
            
            const row = document.createElement('tr');
            
            // 安全地获取会话ID
            const sessionId = session && session.session_id ? String(session.session_id) : '';
            if (!sessionId) {
                return;
            }
            
            // 格式化时间
            const rawTime = session.timestamp || session.completed_time;
            console.log('原始时间戳:', rawTime);
            let timeStr = '-';
            
            if (rawTime) {
                try {
                    const timestamp = new Date(rawTime);
                    console.log('解析后的时间对象:', timestamp);
                    console.log('时间是否有效:', !isNaN(timestamp.getTime()));
                    
                    if (!isNaN(timestamp.getTime())) {
                        timeStr = timestamp.toLocaleString('zh-CN');
                    }
                } catch (timeError) {
                    console.error('时间格式化失败:', timeError);
                }
            }
            
            // 安全地获取配置信息
            const config = session.config || {};
            const symbols = Array.isArray(config.symbols) ? config.symbols : [];
            const timeframes = Array.isArray(config.timeframes) ? config.timeframes : (config.timeframe ? [config.timeframe] : []);
            const selectedAlgorithms = Array.isArray(config.selected_algorithms) ? config.selected_algorithms : [];
            
            // 调试配置信息
            console.log('会话配置:', {
                sessionId: sessionId,
                config: config,
                symbols: symbols,
                timeframes: timeframes,
                selectedAlgorithms: selectedAlgorithms
            });
            
            // 详细调试算法信息
            console.log('算法调试:', {
                rawConfig: session.config,
                selectedAlgorithms: selectedAlgorithms,
                selectedAlgorithmsLength: selectedAlgorithms.length,
                selectedAlgorithmsType: typeof selectedAlgorithms,
                selectedAlgorithmsIsArray: Array.isArray(selectedAlgorithms)
            });
            
            // 安全地获取结果信息
            const results = session.results || {};
            let factorsCount = 0;
            
            try {
                factorsCount = session.factors_count || results.total_factors || results.factors_info?.total_factors || 0;
                // 确保是数字
                factorsCount = parseInt(factorsCount) || 0;
            } catch (countError) {
                console.error('获取因子数量失败:', countError);
                factorsCount = 0;
            }
            
            console.log(`会话 ${index + 1} 因子数量:`, factorsCount);
            
            // 安全地构建HTML
            const safeTimeStr = timeStr.replace(/[<>]/g, '');
            
            const safeSymbols = symbols.map(s => String(s || '').replace(/[<>]/g, '')).join(', ') || '未知';
            const displaySymbols = safeSymbols.length > 20 ? safeSymbols.substring(0, 20) + '...' : safeSymbols;
            
            const safeTimeframes = timeframes.map(t => String(t || '').replace(/[<>]/g, '')).join(', ') || '未知';
            const displayTimeframes = safeTimeframes.length > 15 ? safeTimeframes.substring(0, 15) + '...' : safeTimeframes;
            
            const safeSelectedAlgorithms = selectedAlgorithms.map(a => String(a || '').replace(/[<>]/g, '')).join(', ') || '未知';
            const displaySelectedAlgorithms = safeSelectedAlgorithms.length > 20 ? safeSelectedAlgorithms.substring(0, 20) + '...' : safeSelectedAlgorithms;
            
            const safeFactorsCount = String(factorsCount);
            const safeStatus = String(session.status || '未知').replace(/[<>]/g, '');
            
            const modeRaw = session.mode || config.mode || results.mode || 'unknown';
            const safeMode = String(modeRaw || 'unknown').replace(/[<>]/g, '');
            const modeLabel = safeMode === 'cross_sectional_rl' ? 'RL截面' :
                              safeMode === 'cross_sectional' ? 'GP截面' :
                              safeMode === 'standard' ? '时序' :
                              safeMode === 'cross_sectional_gp' ? 'GP截面' : safeMode;
            const modeBadge = safeMode.includes('rl') ? 'info' : 'primary';

            const canView = (session.status === 'completed' || session.status === 'stopped');
            row.innerHTML = `
                <td>${safeTimeStr}</td>
                <td><span class="badge bg-${modeBadge}">${modeLabel}</span></td>
                <td title="${safeSymbols}">${displaySymbols}</td>
                <td title="${safeTimeframes}">${displayTimeframes}</td>
                <td title="${safeSelectedAlgorithms}">${displaySelectedAlgorithms}</td>
                <td>${safeFactorsCount}</td>
                <td>
                    <span class="badge bg-${session.status === 'completed' ? 'success' : session.status === 'running' ? 'warning' : session.status === 'stopped' ? 'secondary' : 'secondary'}">
                        ${session.status === 'completed' ? '完成' : session.status === 'running' ? '进行中' : session.status === 'stopped' ? '已停止' : '未知'}
                    </span>
                </td>
                <td>
                    ${canView ? 
                        `<button class="btn btn-sm btn-outline-primary me-1" onclick="viewMiningResult('${sessionId}')">
                            <i class="fas fa-eye me-1"></i>查看
                        </button>` : ''
                    }
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteMiningHistory('${sessionId}')">
                        <i class="fas fa-trash me-1"></i>删除
                    </button>
                </td>
            `;
            
            historyTableBody.appendChild(row);
            console.log(`第 ${index + 1} 行添加完成`);
            
        } catch (sessionError) {
            console.error(`处理会话 ${index + 1} 时出错:`, sessionError);
            // 添加错误行
            const errorRow = document.createElement('tr');
            errorRow.innerHTML = `
                <td colspan="8" class="text-center text-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>会话数据错误: ${sessionError.message}
                </td>
            `;
            historyTableBody.appendChild(errorRow);
        }
    });
    
    console.log(`挖掘历史更新完成，共 ${sessions.length} 条记录`);
}

async function deleteMiningHistory(sessionId) {
    try {
        if (!sessionId) return;
        const ok = confirm('确认删除该条挖掘历史？此操作将同时删除对应结果文件，且不可恢复。');
        if (!ok) return;
        const resp = await fetch(`/api/mining/history/delete/${sessionId}`, { method: 'POST' });
        const json = await resp.json();
        if (json && json.success) {
            showAlert('success', '已删除该条挖掘历史');
            await loadMiningHistory();
        } else {
            showAlert('error', `删除失败: ${json && (json.error || json.message) ? (json.error || json.message) : '未知错误'}`);
        }
    } catch (e) {
        showAlert('error', `删除失败: ${e.message}`);
    }
}

async function clearMiningHistory() {
    try {
        const ok = confirm('确认清空全部挖掘历史？此操作将删除所有历史记录与结果文件，且不可恢复。');
        if (!ok) return;
        const resp = await fetch('/api/mining/history/clear', { method: 'POST' });
        const json = await resp.json();
        if (json && json.success) {
            showAlert('success', `已清空历史（删除${json.deleted_count || 0}条）`);
            await loadMiningHistory();
        } else {
            showAlert('error', `清空失败: ${json && (json.error || json.message) ? (json.error || json.message) : '未知错误'}`);
        }
    } catch (e) {
        showAlert('error', `清空失败: ${e.message}`);
    }
}

/**
 * 查看挖掘结果
 */
async function viewMiningResult(sessionId) {
    try {
        console.log('查看挖掘结果:', sessionId);
        
        const response = await fetch(`/api/mining/result/${sessionId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // 获取响应文本
        const responseText = await response.text();
        console.log('挖掘结果响应文本:', responseText);
        
        // 清理NaN值，替换为null
        const cleanedText = responseText.replace(/: NaN/g, ': null');
        console.log('清理后的响应文本:', cleanedText);
        
        let data;
        try {
            // 尝试解析清理后的JSON
            data = JSON.parse(cleanedText);
        } catch (parseError) {
            console.error('JSON解析失败:', parseError);
            console.error('清理后的响应文本:', cleanedText);
            throw new Error(`JSON解析失败: ${parseError.message}`);
        }
        
        console.log('挖掘结果数据:', data);
        
        if (data.success !== false) {
            showMiningResults(data);

            const sid = data.session_id || '';
            if (sid) {
                loadDiffReport(sid);
            }

            const miningResults = document.getElementById('miningResults');
            if (miningResults) {
                miningResults.scrollIntoView({ behavior: 'smooth' });
            }
        } else {
            showAlert('error', data.error || '获取挖掘结果失败');
        }
    } catch (error) {
        console.error('获取挖掘结果失败:', error);
        showAlert('error', `获取挖掘结果失败: ${error.message}`);
    }
}

/**
 * 显示状态信息（替代弹窗）
 */
function showAlert(type, message) {
    console.log(`${type}: ${message}`);
    
    const statusDisplay = document.getElementById('statusDisplay');
    const statusMessage = document.getElementById('statusMessage');
    
    if (statusDisplay && statusMessage) {
        // 设置样式和消息
        statusDisplay.className = `alert alert-${type === 'error' ? 'danger' : type === 'warning' ? 'warning' : type === 'success' ? 'success' : 'info'} mb-3`;
        statusMessage.textContent = message;
        
        // 显示状态区域
        statusDisplay.style.display = 'block';
        
        // 自动隐藏成功和警告消息（5秒后）
        if (type === 'success' || type === 'warning') {
            setTimeout(() => {
                statusDisplay.style.display = 'none';
            }, 5000);
        }
        
        // 错误消息保持显示，直到用户手动关闭或新的状态
        if (type === 'error') {
            // 添加关闭按钮
            if (!statusDisplay.querySelector('.btn-close')) {
                const closeBtn = document.createElement('button');
                closeBtn.className = 'btn-close';
                closeBtn.setAttribute('type', 'button');
                closeBtn.setAttribute('aria-label', 'Close');
                closeBtn.onclick = () => {
                    statusDisplay.style.display = 'none';
                };
                statusDisplay.appendChild(closeBtn);
            }
        }
    }
}

/**
 * 更新子进度条
 */
function updateSubProgress(message, mainProgress) {
    console.log('更新子进度条:', message, mainProgress);
    
    // 显示子进度条容器
    const subProgressContainer = document.getElementById('subProgressContainer');
    if (subProgressContainer) {
        subProgressContainer.style.display = 'block';
    }
    
    // 解析消息，确定当前执行的因子类型
    let currentFactorType = null;
    let currentStatus = 'running';
    let progressPercent = 0;
    
    if (message.includes('机器学习因子') || message.includes('ML因子') || 
        message.includes('机器学习') || message.includes('集成模型') || 
        message.includes('PCA') || message.includes('特征选择') || 
        message.includes('滚动ML') || message.includes('自适应ML')) {
        currentFactorType = 'ml';
    } else if (message.includes('技术因子')) {
        currentFactorType = 'technical';
    } else if (message.includes('统计因子')) {
        currentFactorType = 'statistical';
    } else if (message.includes('高级因子')) {
        currentFactorType = 'advanced';
    }
    
    // 根据主进度计算子进度
    if (mainProgress <= 20) {
        progressPercent = (mainProgress / 20) * 100;
    } else if (mainProgress <= 40) {
        progressPercent = ((mainProgress - 20) / 20) * 100;
    } else if (mainProgress <= 60) {
        progressPercent = ((mainProgress - 40) / 20) * 100;
    } else if (mainProgress <= 80) {
        progressPercent = ((mainProgress - 60) / 20) * 100;
    } else {
        progressPercent = 100;
    }
    
    // 更新所有子进度条
    updateSubProgressItem('ml', currentFactorType === 'ml', progressPercent, message);
    updateSubProgressItem('technical', currentFactorType === 'technical', progressPercent, message);
    updateSubProgressItem('statistical', currentFactorType === 'statistical', progressPercent, message);
    updateSubProgressItem('advanced', currentFactorType === 'advanced', progressPercent, message);
}

/**
 * 更新单个子进度条项目
 */
function updateSubProgressItem(type, isActive, progress, message) {
    const container = document.getElementById(`${type}SubProgress`);
    if (!container) return;
    
    const progressBar = container.querySelector('.sub-progress-fill');
    const status = container.querySelector('.sub-progress-status');
    const details = container.querySelector('.sub-progress-details');
    
    if (isActive) {
        // 显示并激活当前因子类型
        container.style.display = 'block';
        
        // 更新进度条
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.classList.add('animate');
        }
        
        // 更新状态
        if (status) {
            status.textContent = '执行中...';
            status.className = 'sub-progress-status running';
        }
        
        // 更新详细信息
        if (details) {
            details.textContent = message || '正在处理...';
        }
    } else {
        // 隐藏非活跃的因子类型
        container.style.display = 'none';
    }
}

/**
 * 重置子进度条
 */
function resetSubProgress() {
    const subProgressContainer = document.getElementById('subProgressContainer');
    if (subProgressContainer) {
        subProgressContainer.style.display = 'none';
    }
    
    // 重置所有子进度条
    ['ml', 'technical', 'statistical', 'advanced'].forEach(type => {
        const container = document.getElementById(`${type}SubProgress`);
        if (container) {
            const progressBar = container.querySelector('.sub-progress-fill');
            const status = container.querySelector('.sub-progress-status');
            const details = container.querySelector('.sub-progress-details');
            
            if (progressBar) {
                progressBar.style.width = '0%';
                progressBar.classList.remove('animate');
            }
            
            if (status) {
                status.textContent = '准备中...';
                status.className = 'sub-progress-status';
            }
            
            if (details) {
                details.textContent = '初始化...';
            }
        }
    });
}

function showRLResults(data) {
    const miningResults = document.getElementById('miningResults');
    if (!miningResults) return;

    const results = data.results || data;
    const factors = results.factors || {};
    const factorList = Object.entries(factors);

    const totalFactorsEl = document.getElementById('totalFactors');
    const selectedFactorsEl = document.getElementById('selectedFactors');
    const avgICEl = document.getElementById('avgIC');
    const avgReturnEl = document.getElementById('avgReturn');

    if (totalFactorsEl) totalFactorsEl.textContent = factorList.length;
    if (selectedFactorsEl) selectedFactorsEl.textContent = factorList.length;

    if (factorList.length > 0) {
        const icValues = factorList.filter(([, f]) => f.ic_mean != null).map(([, f]) => Math.abs(f.ic_mean || 0));
        if (icValues.length > 0) {
            const avgIC = icValues.reduce((s, v) => s + v, 0) / icValues.length;
            if (avgICEl) avgICEl.textContent = avgIC.toFixed(4);
        } else {
            if (avgICEl) avgICEl.textContent = 'N/A';
        }
        const avgRet = factorList.reduce((s, [, f]) => s + (f.avg_return || 0), 0) / factorList.length;
        if (avgReturnEl) avgReturnEl.textContent = `${(avgRet * 100).toFixed(2)}%`;
    }

    const tableContainer = miningResults.querySelector('.table-responsive');
    if (!tableContainer) return;

    tableContainer.innerHTML = '';

    if (factorList.length === 0) {
        tableContainer.innerHTML = '<div class="alert alert-warning">未发现有效RL截面因子</div>';
        return;
    }

    const table = document.createElement('table');
    table.className = 'table table-sm table-striped table-hover align-middle';
    table.innerHTML = `
        <thead>
            <tr>
                <th style="width:32px;"><input type="checkbox" id="minedSelectAll"></th>
                <th>因子ID</th>
                <th>表达式</th>
                <th>IC均值</th>
                <th>ICIR</th>
                <th>Rank IC</th>
                <th>Rank ICIR</th>
                <th>多空收益</th>
                <th>回测得分</th>
                <th>平均收益</th>
                <th>币种数</th>
                <th>期数</th>
                <th>覆盖率</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');
    const sorted = factorList.sort(([, a], [, b]) => (b.score || 0) - (a.score || 0));

    sorted.forEach(([fid, f]) => {
        const row = document.createElement('tr');
        const icClass = (f.ic_mean || 0) > 0 ? 'text-success' : (f.ic_mean || 0) < 0 ? 'text-danger' : '';
        const scoreClass = (f.score || 0) > 0 ? 'text-success' : (f.score || 0) < 0 ? 'text-danger' : '';
        const expr = (f.expression || '').length > 80 ? (f.expression || '').substring(0, 77) + '...' : (f.expression || '');
        const icDisplay = (f.ic_mean !== null && f.ic_mean !== undefined) ? (f.ic_mean || 0).toFixed(4) : 'N/A';
        const icirDisplay = (f.icir !== null && f.icir !== undefined) ? (f.icir || 0).toFixed(4) : 'N/A';
        const rankIcDisplay = (f.rank_ic_mean !== null && f.rank_ic_mean !== undefined) ? (f.rank_ic_mean || 0).toFixed(4) : 'N/A';
        const rankIcirDisplay = (f.rank_icir !== null && f.rank_icir !== undefined) ? (f.rank_icir || 0).toFixed(4) : 'N/A';
        const lsDisplay = (f.long_short_return !== null && f.long_short_return !== undefined) ? ((f.long_short_return || 0) * 100).toFixed(2) + '%' : 'N/A';
        const coverageRate = (f.coverage_rate !== null && f.coverage_rate !== undefined)
            ? f.coverage_rate
            : ((f.total_periods || 0) > 0 ? (f.n_periods || 0) / f.total_periods : 0);
        row.innerHTML = `
            <td><input type="checkbox" class="minedRowChk" data-factor-id="${fid}"></td>
            <td><code>${fid}</code></td>
            <td title="${escapeHtml(f.expression || '')}"><small>${escapeHtml(expr)}</small></td>
            <td class="${icClass}"><strong>${icDisplay}</strong></td>
            <td>${icirDisplay}</td>
            <td>${rankIcDisplay}</td>
            <td>${rankIcirDisplay}</td>
            <td>${lsDisplay}</td>
            <td class="${scoreClass}"><strong>${(f.score || 0).toFixed(4)}</strong></td>
            <td>${((f.avg_return || 0) * 100).toFixed(2)}%</td>
            <td>${f.n_symbols || 0}</td>
            <td>${f.n_periods || 0}</td>
            <td>${(coverageRate * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });

    tableContainer.appendChild(table);
    renderMinedSaveActions(data.session_id || miningSession, tableContainer);

    const trainHistory = results.training_history || [];
    if (trainHistory.length > 0) {
        const chartDiv = document.createElement('div');
        chartDiv.className = 'mt-4';
        chartDiv.innerHTML = `
            <h5>RL训练曲线</h5>
            <canvas id="rlTrainingChart" height="200"></canvas>
        `;
        tableContainer.parentElement.appendChild(chartDiv);

        setTimeout(() => renderRLTrainingChart(trainHistory), 100);
    }
}

function renderRLTrainingChart(history) {
    const canvas = document.getElementById('rlTrainingChart');
    if (!canvas) return;

    if (canvas._chartInstance) canvas._chartInstance.destroy();

    const sampled = history.filter((_, i) => i % Math.max(1, Math.floor(history.length / 200)) === 0 || i === history.length - 1);

    const ctx = canvas.getContext('2d');
    canvas._chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sampled.map(h => `Step ${h.step}`),
            datasets: [
                {
                    label: '平均奖励',
                    data: sampled.map(h => h.avg_reward || 0),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: '最佳得分',
                    data: sampled.map(h => h.best_score || 0),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    fill: false,
                    tension: 0.3,
                },
                {
                    label: '策略损失',
                    data: sampled.map(h => h.policy_loss || 0),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false,
                    tension: 0.3,
                    borderDash: [5, 5],
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'RL训练过程' }
            },
            scales: {
                y: { title: { display: true, text: '奖励/得分' } },
                y1: {
                    position: 'right',
                    title: { display: true, text: '损失' },
                    grid: { drawOnChartArea: false },
                },
                x: { title: { display: true, text: '训练步数' } }
            }
        }
    });
}

/**
 * 加载算法列表
 */
async function loadAlgorithms() {
    try {
        console.log('开始加载算法列表...');
        const response = await fetch('/api/mining/algorithms');
        const result = await response.json();
        
        if (result.success) {
            console.log(`成功加载 ${result.algorithms.length} 个算法`);
            renderAlgorithmSelector(result.algorithms);
        } else {
            console.error('加载算法失败:', result.error);
            showAlert('error', '加载算法失败: ' + result.error);
        }
    } catch (error) {
        console.error('加载算法异常:', error);
        showAlert('error', '加载算法异常: ' + error.message);
    }
}

/**
 * 渲染算法选择器
 */
function renderAlgorithmSelector(algorithms) {
    const container = document.getElementById('algorithm_selector');
    if (!container) {
        console.warn('算法选择器容器不存在');
        return;
    }
    
    // 按分类组织算法
    const algorithmsByCategory = {};
    algorithms.forEach(algo => {
        const category = algo.category || 'other';
        if (!algorithmsByCategory[category]) {
            algorithmsByCategory[category] = [];
        }
        algorithmsByCategory[category].push(algo);
    });
    
    // 清空容器
    container.innerHTML = '';
    
    // 创建分类标题和算法列表
    Object.keys(algorithmsByCategory).forEach(category => {
        const categoryDiv = document.createElement('div');
        categoryDiv.className = 'algorithm-category mb-3';
        
        const categoryTitle = document.createElement('h6');
        categoryTitle.className = 'text-primary mb-2';
        categoryTitle.textContent = category.toUpperCase();
        categoryDiv.appendChild(categoryTitle);
        
        const algorithmList = document.createElement('div');
        algorithmList.className = 'algorithm-list';
        
        algorithmsByCategory[category].forEach(algo => {
            const algoDiv = document.createElement('div');
            algoDiv.className = 'form-check algorithm-item';
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.className = 'form-check-input algorithm-checkbox';
            checkbox.id = `algo_${algo.id}`;
            checkbox.value = algo.id;
            
            const label = document.createElement('label');
            label.className = 'form-check-label';
            label.htmlFor = `algo_${algo.id}`;
            label.innerHTML = `
                <strong>${algo.name}</strong>
                <small class="text-muted d-block">${algo.description || '无描述'}</small>
            `;
            
            algoDiv.appendChild(checkbox);
            algoDiv.appendChild(label);
            algorithmList.appendChild(algoDiv);
        });
        
        categoryDiv.appendChild(algorithmList);
        container.appendChild(categoryDiv);
    });
    
    console.log('算法选择器渲染完成');
}

/**
 * 获取选中的算法
 */
function getSelectedAlgorithms() {
    const checkboxes = document.querySelectorAll('.algorithm-checkbox:checked');
    return Array.from(checkboxes).map(checkbox => checkbox.value);
}

/**
 * 更新挖掘参数收集，包含算法选择
 */
function collectMiningParams() {
    const params = {
        symbols: getSelectedValues('symbolsSelect'),
        timeframes: getSelectedValues('timeframesSelect'),
        selected_algorithms: getSelectedAlgorithms(),
        start_date: document.getElementById('startDate')?.value || '',
        end_date: document.getElementById('endDate')?.value || '',
        max_factors: parseInt(document.getElementById('maxFactors')?.value) || 15,
        optimization_method: document.getElementById('optimizationMethod')?.value || 'greedy'
    };
    
    // 收集算法参数
    const selectedAlgos = getSelectedAlgorithms();
    selectedAlgos.forEach(algoId => {
        const algoParams = collectAlgorithmParams(algoId);
        if (Object.keys(algoParams).length > 0) {
            params[`${algoId}_params`] = algoParams;
        }
    });
    
    return params;
}

/**
 * 收集特定算法的参数（可扩展）
 */
function collectAlgorithmParams(algoId) {
    return {};
}


// ==================== 截面因子挖掘（GP遗传编程） ====================

function switchMiningMode(mode) {
    currentMiningMode = mode;

    const standardConfig = document.getElementById('standardMiningConfig');
    const csConfig = document.getElementById('crossSectionalConfig');
    const rlConfig = document.getElementById('rlConfig');
    const algoSelector = document.getElementById('algorithm_selector');
    const csAlgoInfo = document.getElementById('csAlgoInfo');
    const standardParams = document.getElementById('standardParams');

    if (mode === 'cross_sectional') {
        if (standardConfig) standardConfig.style.display = 'none';
        if (csConfig) csConfig.style.display = 'block';
        if (rlConfig) rlConfig.style.display = 'none';
        if (algoSelector) algoSelector.style.display = 'none';
        if (csAlgoInfo) csAlgoInfo.style.display = 'block';
        if (standardParams) standardParams.style.display = 'none';
    } else if (mode === 'cross_sectional_rl') {
        if (standardConfig) standardConfig.style.display = 'none';
        if (csConfig) csConfig.style.display = 'none';
        if (rlConfig) rlConfig.style.display = 'block';
        if (algoSelector) algoSelector.style.display = 'none';
        if (csAlgoInfo) csAlgoInfo.style.display = 'none';
        if (standardParams) standardParams.style.display = 'none';
        checkRLTorchStatus();
    } else {
        if (standardConfig) standardConfig.style.display = 'block';
        if (csConfig) csConfig.style.display = 'none';
        if (rlConfig) rlConfig.style.display = 'none';
        if (algoSelector) algoSelector.style.display = 'block';
        if (csAlgoInfo) csAlgoInfo.style.display = 'none';
        if (standardParams) standardParams.style.display = 'block';
    }
}

async function checkRLTorchStatus() {
    const statusEl = document.getElementById('rlTorchStatus');
    if (!statusEl) return;

    try {
        const resp = await fetch('/api/mining/cross_sectional_rl/check_torch');
        const data = await resp.json();

        if (data.torch_available) {
            const deviceInfo = data.cuda_available
                ? `<span class="badge bg-success">CUDA: ${data.device_name}</span>`
                : `<span class="badge bg-warning text-dark">CPU模式（建议使用GPU）</span>`;
            statusEl.innerHTML = `
                <div class="alert alert-success py-2 mb-0">
                    <small>✅ PyTorch ${data.torch_version} ${deviceInfo}</small>
                </div>`;
        } else {
            statusEl.innerHTML = `
                <div class="alert alert-danger py-2 mb-0">
                    <small>❌ PyTorch未安装。请运行: <code>pip install torch</code></small>
                </div>`;
        }
    } catch (e) {
        statusEl.innerHTML = `
            <div class="alert alert-warning py-2 mb-0">
                <small>⚠️ 无法检测PyTorch状态</small>
            </div>`;
    }
}

async function handleRLCrossSectionalMining() {
    try {
        const symbols = getSelectedValues('symbolsSelect');
        const timeframes = getSelectedValues('timeframesSelect');
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        if (symbols.length < 3) {
            showAlert('error', `截面挖掘至少需要3个交易对，当前选择了${symbols.length}个`);
            return;
        }

        if (timeframes.length === 0) {
            showAlert('error', '请选择至少一个时间框架');
            return;
        }

        const dModel = parseInt(document.getElementById('rlDModel')?.value) || 64;
        const nhead = parseInt(document.getElementById('rlNhead')?.value) || 4;
        if (dModel % nhead !== 0) {
            showAlert('error', `d_model(${dModel})必须能被nhead(${nhead})整除，请调整参数`);
            return;
        }

        const exchange = document.getElementById('exchangeSelect')?.value || 'binance';
        const data_source = exchange || 'binance';

        if (timeframes.length > 1) {
            showAlert('warning', `截面RL当前仅使用第一个时间框架：${timeframes[0]}`);
        }

        const rlMinCoverageRaw = parseFloat(document.getElementById('rlMinCoverage')?.value);
        const rlConfig = {
            symbols: symbols,
            timeframe: timeframes[0],
            start_date: startDate || undefined,
            end_date: endDate || undefined,
            data_source: data_source,
            device: document.getElementById('rlDevice')?.value || 'auto',
            batch_size: parseInt(document.getElementById('rlBatchSize')?.value) || 512,
            train_steps: parseInt(document.getElementById('rlTrainSteps')?.value) || 500,
            max_formula_len: parseInt(document.getElementById('rlMaxFormulaLen')?.value) || 16,
            lr: parseFloat(document.getElementById('rlLr')?.value) || 0.001,
            d_model: parseInt(document.getElementById('rlDModel')?.value) || 64,
            nhead: parseInt(document.getElementById('rlNhead')?.value) || 4,
            num_layers: parseInt(document.getElementById('rlNumLayers')?.value) || 2,
            num_loops: parseInt(document.getElementById('rlNumLoops')?.value) || 3,
            entropy_coef: parseFloat(document.getElementById('rlEntropyCoef')?.value) || 0.01,
            use_lord: document.getElementById('rlUseLord')?.checked ?? true,
            trade_size: parseFloat(document.getElementById('rlTradeSize')?.value) || 10000,
            base_fee: parseFloat(document.getElementById('rlBaseFee')?.value) || 0.001,
            max_factors: parseInt(document.getElementById('rlMaxFactors')?.value) || 15,
            max_correlation: parseFloat(document.getElementById('rlMaxCorrelation')?.value) || 0.7,
            min_coverage: Number.isFinite(rlMinCoverageRaw) ? rlMinCoverageRaw : 0.2,
        };

        const rlIncludeExtras = [];
        if (document.getElementById('rlExtrasBasis')?.checked) rlIncludeExtras.push('basis');
        if (document.getElementById('rlExtrasMetrics')?.checked) rlIncludeExtras.push('metrics');
        if (document.getElementById('rlExtrasFunding')?.checked) rlIncludeExtras.push('funding');
        rlConfig.include_extras = rlIncludeExtras;

        console.log('RL截面挖掘配置:', rlConfig);

        updateStartButton(true);
        showWaitingState();

        const response = await fetch('/api/mining/cross_sectional_rl/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rlConfig)
        });

        const result = await response.json();
        console.log('RL截面挖掘启动结果:', result);

        if (result.success) {
            miningSession = result.session_id;
            showAlert('info', `RL截面挖掘已启动！${symbols.length}个币种，BS=${rlConfig.batch_size}，${rlConfig.train_steps}步训练`);
            showMiningProgress();
            startProgressMonitoring(result.session_id);
        } else {
            showAlert('error', result.error || 'RL截面挖掘启动失败');
            updateStartButton(false);
        }

    } catch (error) {
        console.error('RL截面挖掘失败:', error);
        showAlert('error', `RL截面挖掘失败: ${error.message}`);
        updateStartButton(false);
    }
}

async function handleCrossSectionalMining() {
    try {
        const symbols = getSelectedValues('symbolsSelect');
        const timeframes = getSelectedValues('timeframesSelect');
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        if (symbols.length < 3) {
            showAlert('error', `截面挖掘至少需要3个交易对，当前选择了${symbols.length}个`);
            return;
        }

        if (timeframes.length === 0) {
            showAlert('error', '请选择至少一个时间框架');
            return;
        }

        const exchange = document.getElementById('exchangeSelect')?.value || 'binance';
        const data_source = exchange || 'binance';

        if (timeframes.length > 1) {
            showAlert('warning', `截面GP当前仅使用第一个时间框架：${timeframes[0]}`);
        }

        const gpMinCoverageRaw = parseFloat(document.getElementById('csMinCoverage')?.value);
        const gpIncludeExtras = [];
        if (document.getElementById('csExtrasBasis')?.checked) gpIncludeExtras.push('basis');
        if (document.getElementById('csExtrasMetrics')?.checked) gpIncludeExtras.push('metrics');
        if (document.getElementById('csExtrasFunding')?.checked) gpIncludeExtras.push('funding');
        const gpConfig = {
            symbols: symbols,
            timeframe: timeframes[0],
            start_date: startDate || undefined,
            end_date: endDate || undefined,
            data_source: data_source,
            population_size: parseInt(document.getElementById('csPopulationSize')?.value) || 200,
            max_generations: parseInt(document.getElementById('csMaxGenerations')?.value) || 30,
            max_depth: parseInt(document.getElementById('csMaxDepth')?.value) || 5,
            crossover_rate: parseFloat(document.getElementById('csCrossoverRate')?.value) || 0.7,
            mutation_rate: parseFloat(document.getElementById('csMutationRate')?.value) || 0.2,
            min_ic: parseFloat(document.getElementById('csMinIC')?.value) || 0.02,
            min_ir: parseFloat(document.getElementById('csMinIR')?.value) || 0.1,
            max_factors: parseInt(document.getElementById('csMaxFactors')?.value) || 15,
            max_correlation: parseFloat(document.getElementById('csMaxCorrelation')?.value) || 0.7,
            min_coverage: Number.isFinite(gpMinCoverageRaw) ? gpMinCoverageRaw : 0.2,
            include_extras: gpIncludeExtras,
        };

        console.log('截面挖掘配置:', gpConfig);

        updateStartButton(true);
        showWaitingState();

        const response = await fetch('/api/mining/cross_sectional/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(gpConfig)
        });

        const result = await response.json();
        console.log('截面挖掘启动结果:', result);

        if (result.success) {
            miningSession = result.session_id;
            showAlert('info', `截面挖掘已启动！${symbols.length}个币种，种群${gpConfig.population_size}，${gpConfig.max_generations}代进化`);
            showMiningProgress();
            startProgressMonitoring(result.session_id);
        } else {
            showAlert('error', result.error || '截面挖掘启动失败');
            updateStartButton(false);
        }

    } catch (error) {
        console.error('截面挖掘失败:', error);
        showAlert('error', `截面挖掘失败: ${error.message}`);
        updateStartButton(false);
    }
}

function showCrossSectionalResults(data) {
    const miningResults = document.getElementById('miningResults');
    if (!miningResults) return;

    const results = data.results || data;
    const factors = results.factors || {};
    const factorList = Object.entries(factors);

    const totalFactorsEl = document.getElementById('totalFactors');
    const selectedFactorsEl = document.getElementById('selectedFactors');
    const avgICEl = document.getElementById('avgIC');
    const avgReturnEl = document.getElementById('avgReturn');

    if (totalFactorsEl) totalFactorsEl.textContent = factorList.length;
    if (selectedFactorsEl) selectedFactorsEl.textContent = factorList.length;

    if (factorList.length > 0) {
        const avgIC = factorList.reduce((s, [, f]) => s + Math.abs(f.ic_mean || 0), 0) / factorList.length;
        const avgRet = factorList.reduce((s, [, f]) => s + (f.long_short_return || 0), 0) / factorList.length;
        if (avgICEl) avgICEl.textContent = avgIC.toFixed(4);
        if (avgReturnEl) avgReturnEl.textContent = `${(avgRet * 100).toFixed(2)}%`;
    }

    const tableContainer = miningResults.querySelector('.table-responsive');
    if (!tableContainer) return;

    tableContainer.innerHTML = '';

    if (factorList.length === 0) {
        tableContainer.innerHTML = '<div class="alert alert-warning">未发现有效截面因子</div>';
        return;
    }

    const table = document.createElement('table');
    table.className = 'table table-sm table-striped table-hover align-middle';
    table.innerHTML = `
        <thead>
            <tr>
                <th style="width:32px;"><input type="checkbox" id="minedSelectAll"></th>
                <th>因子ID</th>
                <th>表达式</th>
                <th>方向</th>
                <th>IC均值</th>
                <th>ICIR</th>
                <th>Fitness</th>
                <th>Rank IC</th>
                <th>Rank ICIR</th>
                <th>多空收益</th>
                <th>币种数</th>
                <th>期数</th>
                <th>覆盖率</th>
            </tr>
        </thead>
        <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');
    const sorted = factorList.sort(([, a], [, b]) => {
        const fa = Math.abs(a.fitness ?? a.icir ?? 0);
        const fb = Math.abs(b.fitness ?? b.icir ?? 0);
        return fb - fa;
    });

    sorted.forEach(([fid, f]) => {
        const row = document.createElement('tr');
        const icClass = (f.ic_mean || 0) > 0 ? 'text-success' : (f.ic_mean || 0) < 0 ? 'text-danger' : '';
        const isNegative = (f.direction === 'negative') || ((f.ic_mean || 0) < 0);
        const directionLabel = isNegative ? '反向' : '正向';
        const directionBadge = isNegative ? 'bg-warning text-dark' : 'bg-success';
        const expr = (f.expression || '').length > 80 ? (f.expression || '').substring(0, 77) + '...' : (f.expression || '');
        const coverageRate = (f.coverage_rate !== null && f.coverage_rate !== undefined)
            ? f.coverage_rate
            : ((f.total_periods || 0) > 0 ? (f.n_periods || 0) / f.total_periods : 0);
        row.innerHTML = `
            <td><input type="checkbox" class="minedRowChk" data-factor-id="${fid}"></td>
            <td><code>${fid}</code></td>
            <td title="${escapeHtml(f.expression || '')}"><small>${escapeHtml(expr)}</small></td>
            <td><span class="badge ${directionBadge}">${directionLabel}</span></td>
            <td class="${icClass}"><strong>${(f.ic_mean || 0).toFixed(4)}</strong></td>
            <td>${(f.icir || 0).toFixed(4)}</td>
            <td>${(f.fitness || 0).toFixed(4)}</td>
            <td>${(f.rank_ic_mean || 0).toFixed(4)}</td>
            <td>${(f.rank_icir || 0).toFixed(4)}</td>
            <td>${((f.long_short_return || 0) * 100).toFixed(2)}%</td>
            <td>${f.n_symbols || 0}</td>
            <td>${f.n_periods || 0}</td>
            <td>${(coverageRate * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });

    tableContainer.appendChild(table);
    renderMinedSaveActions(data.session_id || miningSession, tableContainer);

    const genStats = results.generation_stats || [];
    if (genStats.length > 0) {
        const parent = tableContainer.parentElement;
        if (parent) {
            const existing = document.getElementById('evolutionChartWrap');
            if (existing) existing.remove();

            const wrap = document.createElement('div');
            wrap.id = 'evolutionChartWrap';
            wrap.className = 'mt-4';
            wrap.innerHTML = `
                <details id="evolutionChartDetails">
                    <summary class="fw-bold">进化曲线（点击展开）</summary>
                    <div class="evolution-chart-container mt-2">
                        <canvas id="evolutionChart" height="180"></canvas>
                    </div>
                    <div class="text-muted small mt-1">为避免卡顿，曲线将按需渲染并自动抽样显示</div>
                </details>
            `;
            parent.appendChild(wrap);

            const details = document.getElementById('evolutionChartDetails');
            if (details) {
                details.addEventListener('toggle', () => {
                    if (details.open) {
                        setTimeout(() => renderEvolutionChart(genStats), 0);
                    }
                }, { once: true });
            }
        }
    }
}

function renderEvolutionChart(genStats) {
    const canvas = document.getElementById('evolutionChart');
    if (!canvas) return;

    if (canvas._chartInstance) canvas._chartInstance.destroy();

    const sampled = genStats.filter((_, i) => i % Math.max(1, Math.floor(genStats.length / 200)) === 0 || i === genStats.length - 1);

    const ctx = canvas.getContext('2d');
    canvas._chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sampled.map(g => `Gen ${g.generation}`),
            datasets: [
                {
                    label: '最佳IC',
                    data: sampled.map(g => g.best_ic || 0),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: '最佳ICIR',
                    data: sampled.map(g => g.best_icir || 0),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    fill: true,
                    tension: 0.3,
                },
                {
                    label: '平均适应度',
                    data: sampled.map(g => g.avg_fitness || 0),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    fill: false,
                    tension: 0.3,
                    borderDash: [5, 5],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'GP进化过程' }
            },
            scales: {
                y: { title: { display: true, text: '指标值' } },
                x: { title: { display: true, text: '进化代数' } }
            }
        }
    });
}
