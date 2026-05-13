import os
import re

template_dir = r"d:\PythonProject\FactorMiner\webui\templates"

for root, dirs, files in os.walk(template_dir):
    for file in files:
        if file.endswith(".html"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 1. Update tables: add table-hover, align-middle, text-center
            # Match <table class="..."> and inject missing classes
            def update_table_class(match):
                class_str = match.group(1)
                classes = class_str.split()
                if "table" not in classes:
                    classes.insert(0, "table")
                if "table-hover" not in classes:
                    classes.append("table-hover")
                if "align-middle" not in classes:
                    classes.append("align-middle")
                if "text-center" not in classes:
                    classes.append("text-center")
                # Remove duplicate classes
                classes = list(dict.fromkeys(classes))
                return f'<table class="{" ".join(classes)}"'

            content = re.sub(r'<table\s+class="([^"]+)"', update_table_class, content)

            # If there's a table without class attribute
            content = re.sub(r'<table(?!\s+class=)([^>]*)>', r'<table class="table table-hover align-middle text-center"\1>', content)

            # 2. Update h2 to have page-title (if not already)
            # Find the first <h2> that doesn't have accordion-header
            # Actually, just find <h2> tag with an emoji
            def update_h2(match):
                class_attr = match.group(1)
                h2_content = match.group(2)
                
                # Skip if it's an accordion header
                if class_attr and 'accordion-header' in class_attr:
                    return match.group(0)
                    
                classes = []
                if class_attr:
                    m = re.search(r'class="([^"]*)"', class_attr)
                    if m:
                        classes = m.group(1).split()
                
                if 'page-title' not in classes:
                    classes.append('page-title')
                if 'mb-4' not in classes and 'mb-3' not in classes and 'mb-2' not in classes:
                    classes.append('mb-4')
                
                # Clean up existing class attribute
                class_attr_new = f' class="{" ".join(classes)}"'
                rest_attrs = re.sub(r'\s*class="[^"]*"', '', class_attr or '')
                
                return f'<h2{class_attr_new}{rest_attrs}>{h2_content}</h2>'

            content = re.sub(r'<h2([^>]*)>(.*?)</h2>', update_h2, content, flags=re.DOTALL)
            
            # 3. Add shadow-sm border-0 to cards for a cleaner look
            def update_card(match):
                class_str = match.group(1)
                classes = class_str.split()
                if 'shadow-sm' not in classes:
                    classes.append('shadow-sm')
                if 'border-0' not in classes:
                    classes.append('border-0')
                return f'<div class="{" ".join(classes)}"'

            content = re.sub(r'<div\s+class="([^"]*\bcard\b[^"]*)"', update_card, content)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

print("Templates optimized!")
