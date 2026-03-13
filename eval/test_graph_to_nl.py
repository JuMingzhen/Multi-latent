#!/usr/bin/env python3
"""
测试图转自然语言功能
"""

import json
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import graph_to_natural_language

def test_graph_to_nl():
    """测试图转自然语言"""
    # 读取一个测试样本
    test_file = Path(__file__).parent.parent / "data" / "easy_test.jsonl"
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        line = f.readline()
        if line:
            record = json.loads(line)
            
            print("="*60)
            print("Original Record:")
            print("="*60)
            print(f"Question: {record.get('question', 'N/A')}")
            print(f"Answer: {record.get('answer', 'N/A')}")
            print(f"Edges: {record.get('edges', [])[:5]}...")  # 只显示前5条边
            print(f"Names: {dict(list(record.get('names', {}).items())[:10])}...")  # 只显示前10个节点
            
            print("\n" + "="*60)
            print("Converted Natural Language:")
            print("="*60)
            nl_text = graph_to_natural_language(record)
            print(nl_text)
            print("="*60)

if __name__ == "__main__":
    test_graph_to_nl()

