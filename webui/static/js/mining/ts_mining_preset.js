/**
 * 时序挖掘页：锁定为 standard 模式
 */
(function () {
    function applyPreset() {
        try {
            if (typeof switchMiningMode === 'function') {
                switchMiningMode('standard');
            }
        } catch (e) {
            console.warn('ts_mining_preset apply failed:', e);
        }

        // 隐藏模式选择器（避免用户切到截面模式）
        try {
            const modeRadios = document.querySelectorAll('input[name="miningMode"]');
            modeRadios.forEach(r => {
                r.disabled = true;
            });
            const modeStandard = document.getElementById('modeStandard');
            if (modeStandard) modeStandard.disabled = false;
        } catch (e) {
            console.warn('ts_mining_preset lock radios failed:', e);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyPreset);
    } else {
        applyPreset();
    }
})();

