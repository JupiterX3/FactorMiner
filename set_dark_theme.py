import re

file_path = r"d:\PythonProject\FactorMiner\webui\templates\base.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

dark_style = """
        /* 全局现代排版美化覆盖 - 极客暗黑主题 (Dark Mode) */
        body {
            background: #0f172a !important; /* 深空蓝黑背景 */
            color: #e2e8f0 !important; /* 浅灰白文字 */
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }
        
        .text-muted { color: #94a3b8 !important; }
        .bg-light { background-color: #1e293b !important; }
        
        /* 导航栏 Glassmorphism 效果 (暗色版) */
        .navbar {
            background: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5) !important;
        }
        .navbar-brand {
            color: #818cf8 !important; /* 亮紫色Logo */
            font-size: 1.35rem;
            letter-spacing: -0.5px;
        }
        .navbar-nav .nav-link {
            color: #cbd5e1 !important; /* 明亮的灰白 */
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        .navbar-nav .nav-link:hover, .navbar-nav .dropdown-menu .dropdown-item:hover {
            color: #ffffff !important;
            background-color: rgba(99, 102, 241, 0.15);
        }
        
        .dropdown-menu {
            background: rgba(30, 41, 59, 0.95) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
        }
        .dropdown-item {
            color: #cbd5e1 !important;
        }
        
        /* 卡片高级感优化 (暗黑版) */
        .card {
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
            border-radius: 14px !important;
            background: #1e293b !important; /* 稍浅的暗色卡片 */
            color: #e2e8f0 !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
        }
        .card-header {
            background: transparent !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
            padding: 1.25rem 1.25rem 1rem !important;
            color: #f8fafc !important;
        }
        
        /* 按钮与表单优化 (暗黑版) */
        .btn {
            border-radius: 8px !important;
            font-weight: 500 !important;
            letter-spacing: 0.3px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
            border: none !important;
            color: #ffffff !important;
        }
        .btn-primary:hover {
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5) !important;
            transform: translateY(-1px);
        }
        .btn-outline-secondary { color: #cbd5e1 !important; border-color: #475569 !important; }
        .btn-outline-secondary:hover { background: #334155 !important; }
        
        .form-control, .form-select {
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        }
        .form-control:focus, .form-select:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
            background-color: #0f172a !important;
            color: #ffffff !important;
        }
        
        /* 表格微调 (暗黑版) */
        .table {
            color: #e2e8f0 !important;
        }
        .table th, .table td {
            border-color: rgba(255, 255, 255, 0.05) !important;
            color: #e2e8f0 !important;
        }
        .table-hover tbody tr:hover td {
            background-color: rgba(99, 102, 241, 0.1) !important;
            color: #ffffff !important;
        }
        
        /* 页脚 */
        .footer {
            background-color: #0f172a !important;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* 标题等 */
        .page-title, .content-title, .card-title, h1, h2, h3, h4, h5 {
            color: #f8fafc !important;
        }
"""

content = re.sub(r'/\* 全局现代排版美化覆盖 \(Global Aesthetic Overrides\) \*/.*?/\* 表格微调 \*/.*?\}', dark_style.strip(), content, flags=re.DOTALL)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated to dark theme.")
