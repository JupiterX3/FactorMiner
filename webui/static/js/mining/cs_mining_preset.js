/**
 * 截面挖掘页：锁定为 cross_sectional（用户可在 GP / RL 间切换）
 */
(function () {
    function applyPreset() {
        try {
            if (typeof switchMiningMode === 'function') {
                switchMiningMode('cross_sectional');
            }
        } catch (e) {
            console.warn('cs_mining_preset apply failed:', e);
        }

        // 禁用 standard 选项，只保留 cross_sectional / cross_sectional_rl
        try {
            const modeStandard = document.getElementById('modeStandard');
            if (modeStandard) modeStandard.disabled = true;
        } catch (e) {
            console.warn('cs_mining_preset lock standard failed:', e);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyPreset);
    } else {
        applyPreset();
    }
})();

