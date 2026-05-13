import re

file_path = r"d:\PythonProject\FactorMiner\webui\static\css\style.css"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace body background
content = re.sub(
    r'body\s*\{[^}]*background:\s*linear-gradient\(135deg,\s*#667eea\s*0%,\s*#764ba2\s*100%\);[^}]*\}',
    r"body {\n    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;\n    line-height: 1.5;\n    color: var(--text-primary);\n    background: #f4f7f8;\n    min-height: 100vh;\n}",
    content
)

# Replace navbar style
content = re.sub(
    r'\.navbar\s*\{[^}]*background:\s*linear-gradient[^}]*\}',
    r".navbar {\n    background: rgba(255, 255, 255, 0.9) !important;\n    backdrop-filter: blur(12px);\n    border-bottom: 1px solid rgba(0,0,0,0.05);\n    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);\n}",
    content
)

# Replace navbar-brand
content = re.sub(
    r'\.navbar-brand\s*\{[^}]*\}',
    r".navbar-brand {\n    font-weight: 700;\n    font-size: 1.25rem;\n    color: var(--primary-color) !important;\n    text-decoration: none;\n}",
    content
)

# Replace navbar-nav nav-link
content = re.sub(
    r'\.navbar-nav \.nav-link\s*\{[^}]*\}',
    r".navbar-nav .nav-link {\n    color: var(--text-primary) !important;\n    text-decoration: none;\n    padding: 0.5rem 1rem;\n    font-weight: 500;\n    font-size: 0.95rem;\n    transition: all 0.2s ease;\n    border-radius: 6px;\n}",
    content
)

# Replace navbar-nav nav-link:hover
content = re.sub(
    r'\.navbar-nav \.nav-link:hover\s*\{[^}]*\}',
    r".navbar-nav .nav-link:hover {\n    color: var(--primary-color) !important;\n    background-color: rgba(102, 126, 234, 0.1);\n}",
    content
)

# Replace card-header
content = re.sub(
    r'\.card-header\s*\{[^}]*background:\s*linear-gradient\(135deg,\s*var\(--bg-secondary\),\s*var\(--bg-card\)\);[^}]*\}',
    r".card-header {\n    background: transparent;\n    border-bottom: 1px solid var(--border-color);\n    border-radius: 12px 12px 0 0;\n    padding: 1rem 1.25rem;\n}",
    content
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("CSS optimized!")
