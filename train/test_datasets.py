#!/usr/bin/env python3
"""
测试不同数据集类型
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from train.dataset import AnswerOnlySFTDataset, CoTSFTDataset

def test_answer_only_dataset():
    """测试AnswerOnly数据集"""
    print("="*60)
    print("Testing AnswerOnlySFTDataset")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = AnswerOnlySFTDataset(
        data_file="data/easy_test.jsonl",
        tokenizer=tokenizer,
        max_length=512,
        max_samples=3,
        system_prompt="You are a logical reasoning assistant.",
        instruction_template="{system_prompt}\n\n{input}\n\nAnswer:"
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 测试一个样本
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Input IDs length: {len(sample['input_ids'])}")
    print(f"  Labels length: {len(sample['labels'])}")
    
    # 检查labels
    valid_labels = [l for l in sample['labels'] if l != -100]
    print(f"  Valid labels (not -100): {len(valid_labels)}")
    print(f"  Valid label tokens: {tokenizer.decode(valid_labels[:20], skip_special_tokens=True)}")
    
    print("\n✓ AnswerOnlySFTDataset test passed!")


def test_cot_dataset():
    """测试CoT数据集"""
    print("\n" + "="*60)
    print("Testing CoTSFTDataset")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dataset = CoTSFTDataset(
        data_file="data/easy_test.jsonl",
        tokenizer=tokenizer,
        max_length=512,
        max_samples=3,
        system_prompt="You are a logical reasoning assistant.",
        instruction_template="{system_prompt}\n\n{input}\n\nAnswer:",
        cot_prefix="Let me think step by step:"
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # 测试一个样本
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Input IDs length: {len(sample['input_ids'])}")
    print(f"  Labels length: {len(sample['labels'])}")
    
    # 检查labels
    valid_labels = [l for l in sample['labels'] if l != -100]
    print(f"  Valid labels (not -100): {len(valid_labels)}")
    print(f"  Valid label tokens: {tokenizer.decode(valid_labels[:50], skip_special_tokens=True)}")
    
    # 查看原始样本
    original_sample = dataset.samples[0]
    print(f"\n  Original output text: {original_sample['output'][:200]}...")
    
    print("\n✓ CoTSFTDataset test passed!")


if __name__ == "__main__":
    test_answer_only_dataset()
    test_cot_dataset()
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

