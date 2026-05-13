#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactorMiner WebUI 启动脚本
"""

import sys
import os
import signal
from pathlib import Path

# 禁用Flask文件监控，避免因factorlib文件更新而重启
os.environ['FLASK_ENV'] = 'development'

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from webui.app import app
from config import WEBUI_CONFIG

_sigint_count = 0


def _sigint_handler(signum, frame):
    global _sigint_count
    _sigint_count += 1

    if _sigint_count == 1:
        print("\n⚠️  正在取消评估并停止服务... (再按一次 Ctrl+C 强制退出)")
        try:
            from webui.routes.factors import trigger_global_shutdown
            trigger_global_shutdown()
        except Exception:
            pass
        raise KeyboardInterrupt
    else:
        print("\n强制退出...")
        os._exit(1)


def main():
    """启动WebUI"""
    signal.signal(signal.SIGINT, _sigint_handler)

    print("=== FactorMiner WebUI 启动 ===")
    print(f"访问地址: http://{WEBUI_CONFIG['host']}:{WEBUI_CONFIG['port']}")
    print(f"调试模式: {'开启' if WEBUI_CONFIG['debug'] else '关闭'}")
    print("按 Ctrl+C 停止服务")
    print()
    
    try:
        app.run(
            host=WEBUI_CONFIG['host'],
            port=WEBUI_CONFIG['port'],
            debug=WEBUI_CONFIG['debug'],
            use_reloader=False  # 禁用自动重载
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as e:
        print(f"启动失败: {e}")


if __name__ == "__main__":
    main()