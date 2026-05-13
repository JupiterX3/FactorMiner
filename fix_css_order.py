import re

file_path = r"d:\PythonProject\FactorMiner\webui\templates\base.html"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Extract CSS links from the bottom
bootstrap_css = '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">'
fa_css = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.2/css/all.min.css">'

# Remove them from the bottom
content = content.replace('    <!-- Bootstrap CSS -->\n    ' + bootstrap_css + '\n', '')
content = content.replace('    <!-- Font Awesome -->\n    ' + fa_css + '\n', '')

# Insert them into the <head> BEFORE the <style> tags
head_insert = f"""
    <!-- Bootstrap CSS -->
    {bootstrap_css}
    <!-- Font Awesome -->
    {fa_css}
"""
content = content.replace('    <!-- 自定义样式 -->', head_insert + '    <!-- 自定义样式 -->')

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("CSS links moved to head.")
