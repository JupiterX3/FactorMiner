"""
WebUI 配置文件
"""

import os
from pathlib import Path

PROXY_CONFIG = {
    'http': os.environ.get('HTTP_PROXY', 'http://127.0.0.1:7897'),
    'https': os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:7897'),
}

if PROXY_CONFIG['http'] and not os.environ.get('HTTP_PROXY'):
    os.environ['HTTP_PROXY'] = PROXY_CONFIG['http']
    os.environ['HTTPS_PROXY'] = PROXY_CONFIG['https']

IS_PRODUCTION = os.environ.get('FLASK_ENV') == 'production'

WEBUI_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 8080)),
    'debug': not IS_PRODUCTION,
    'secret_key': os.environ.get('SECRET_KEY', 'your-secret-key-here'),
    'upload_folder': Path(__file__).parent.parent / "webui" / "uploads",
    'data_dir': Path(__file__).parent.parent / "data",
    'max_content_length': 16 * 1024 * 1024,
    'threaded': True,
}

WEBUI_CONFIG['upload_folder'].mkdir(exist_ok=True)

FRONTEND_CONFIG = {
    'title': 'FactorMiner - 量化因子挖掘平台',
    'version': '1.0.0',
    'description': '专业的量化因子挖掘、评估和优化平台',
    'author': 'FactorMiner Team'
}

API_CONFIG = {
    'version': 'v1',
    'prefix': '/api/v1',
    'rate_limit': '100 per minute',
    'cors_origins': ['http://localhost:3000', 'http://127.0.0.1:3000']
}

if IS_PRODUCTION:
    WEBUI_CONFIG['debug'] = False
    if WEBUI_CONFIG['secret_key'] == 'your-secret-key-here':
        import warnings
        warnings.warn(
            "生产环境请设置 SECRET_KEY 环境变量！",
            RuntimeWarning
        ) 