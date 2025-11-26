#!/usr/bin/env python3
"""
Quick fix for broken YAML files with embedded JSON
"""

import re
from pathlib import Path

def fix_yaml_file(yaml_path: Path):
    """Fix YAML file by converting embedded JSON to comments"""
    
    if not yaml_path.exists():
        return False
    
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Check if it has the JSON formatting issue
    if '{\n  "' in content:
        print(f"Fixing {yaml_path}...")
        
        # Replace JSON block with YAML comments
        pattern = r'# Conversion statistics:\n\{\n(.*?)\n\}'
        
        def json_to_comments(match):
            json_block = match.group(1)
            comments = ["# Conversion statistics:"]
            
            # Extract key-value pairs from JSON
            for line in json_block.split('\n'):
                line = line.strip()
                if '"' in line and ':' in line:
                    # Extract "key": value
                    key = line.split('"')[1]
                    value = line.split(':')[1].strip().rstrip(',')
                    comments.append(f"# - {key}: {value} samples")
            
            return '\n'.join(comments)
        
        fixed_content = re.sub(pattern, json_to_comments, content, flags=re.DOTALL)
        
        # Write fixed content
        with open(yaml_path, 'w') as f:
            f.write(fixed_content)
        
        print(f"  Fixed YAML formatting")
        return True
    else:
        print(f"  No issues found in {yaml_path}")
        return False

def main():
    """Fix all YAML files in consolidated datasets"""
    print("YAML FILE FIXER")
    print("=" * 30)
    
    datasets_dir = Path("consolidated_datasets")
    
    if not datasets_dir.exists():
        print("No consolidated_datasets directory found")
        return
    
    yaml_files = list(datasets_dir.rglob("data.yaml"))
    
    if not yaml_files:
        print("No YAML files found")
        return
    
    print(f"Found {len(yaml_files)} YAML files")
    
    fixed_count = 0
    for yaml_file in yaml_files:
        if fix_yaml_file(yaml_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} YAML files")
    
    if fixed_count > 0:
        print("You can now run training on the fixed datasets!")

if __name__ == "__main__":
    main()