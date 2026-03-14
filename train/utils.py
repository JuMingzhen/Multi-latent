#!/usr/bin/env python3
"""
训练工具函数
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def graph_to_natural_language(record: Dict) -> Tuple[str, str]:
    """
    将图结构转换为自然语言描述
    
    返回: (context, question)
    """
    edges = record["edges"]
    names = {int(k): v for k, v in record["names"].items()}
    
    # 识别实体节点（没有父节点的节点）
    nodes = set(names.keys())
    parents = set()
    for u, v in edges:
        parents.add(v)
    entity_nodes = nodes - parents
    
    statements = []
    
    # 按边的顺序处理，保持一致性
    for edge in edges:
        u, v = edge[0], edge[1]
        u_name = names.get(u, str(u))
        v_name = names.get(v, str(v))
        
        if u in entity_nodes:
            # 实体到概念的关系
            statements.append(f"{u_name} is a {v_name}.")
        else:
            # 概念到概念的关系
            statements.append(f"Every {u_name} is a {v_name}.")
    
    # 组合所有陈述
    context = " ".join(statements)
    
    # 获取问题
    question = record.get("question", "")
    
    return context, question


def format_prompt(
    context: str,
    question: str,
    entity_name: str,
    answer: str,
    system_prompt: str,
    instruction_template: str = "{system_prompt}\n\n{input}\n\nAnswer:"
) -> Tuple[str, str]:
    """
    格式化训练样本为输入-输出对
    
    返回: (input_text, output_text)
    """
    # 组合输入
    input_text = f"{context} {question}".strip()
    
    # 如果有系统提示词，添加到输入中
    if system_prompt:
        if "{system_prompt}" in instruction_template and "{input}" in instruction_template:
            input_text = instruction_template.format(
                system_prompt=system_prompt,
                input=input_text
            )
        else:
            input_text = f"{system_prompt}\n\n{input_text}"
    
    # 输出就是答案
    output_text = f"{entity_name} is a {answer}"
    
    return input_text, output_text


def load_jsonl(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

