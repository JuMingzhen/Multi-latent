#!/usr/bin/env python3
"""
数据集处理模块
支持多种SFT训练策略：只有答案、CoT（思维链）等
"""

import json
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch

from utils import graph_to_natural_language, format_prompt, load_jsonl


def path_to_cot_text(path: List[int], names: Dict[int, str], answer: str) -> str:
    """
    将路径转换为思维链文本
    
    Args:
        path: 节点路径，例如 [0, 2, 15, 19]
        names: 节点ID到名称的映射
        answer: 最终答案
    
    Returns:
        思维链文本，例如 "Daniel is a korpal. Every korpal is a sherpus. Every sherpus is a fompus. Daniel is a fompus."
    """
    if len(path) < 2:
        return answer
    
    # 获取路径上的节点名称
    path_names = [names.get(node_id, str(node_id)) for node_id in path]
    
    # 构建思维链
    cot_statements = []
    for i in range(len(path_names) - 1):
        current = path_names[i]
        next_node = path_names[i + 1]
        
        if i == 0:
            # 第一个节点是实体
            cot_statements.append(f"{current} is a {next_node}.")
        else:
            # 后续节点是概念
            cot_statements.append(f"Every {current} is a {next_node}.")
    
    # 添加结论
    entity = path_names[0]
    final_concept = path_names[-1]
    conclusion = f"{entity} is a {final_concept}."
    
    cot_text = " ".join(cot_statements) + " " + conclusion
    
    return cot_text


class AnswerOnlySFTDataset(Dataset):
    """
    只有答案的SFT数据集
    只对答案部分计算loss，输入部分不计算loss
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        system_prompt: str = "",
        instruction_template: str = "{system_prompt}\n\n{input}\n\nAnswer:"
    ):
        """
        初始化数据集
        
        Args:
            data_file: JSONL数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            max_samples: 最大样本数（None表示全部）
            system_prompt: 系统提示词
            instruction_template: 指令模板
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.instruction_template = instruction_template
        
        # 加载原始数据
        raw_data = load_jsonl(data_file, max_samples)
        
        # 转换为训练样本
        self.samples = []
        for record in raw_data:
            try:
                # 图转自然语言
                context, question = graph_to_natural_language(record)
                answer = record.get("answer", "")
                
                if not answer:
                    continue
                
                # 格式化输入输出
                input_text, output_text = format_prompt(
                    context=context,
                    entity_name=record.get("names", {}).get(0, ""),
                    question=question,
                    answer=answer,
                    system_prompt=system_prompt,
                    instruction_template=instruction_template
                )
                
                self.samples.append({
                    "input": input_text,
                    "output": output_text,
                    "full_text": f"{input_text}{output_text}"
                })
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} samples from {data_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_text = sample["input"]
        output_text = sample["output"]
        
        # 分别对input和output进行编码
        input_encoded = self.tokenizer(
            input_text,
            add_special_tokens=True,
            return_tensors=None
        )
        input_ids_input = input_encoded["input_ids"]
        
        # 对output编码（不添加特殊token，因为已经包含在input的末尾）
        output_encoded = self.tokenizer(
            output_text,
            add_special_tokens=False,
            return_tensors=None
        )
        output_ids = output_encoded["input_ids"]
        
        # 组合input和output
        input_ids = input_ids_input + output_ids
        
        # 截断到最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # 创建labels：input部分设为-100（不计算loss），output部分保留
        input_len = len(input_ids_input)
        labels = [-100] * min(input_len, len(input_ids)) + input_ids[min(input_len, len(input_ids)):]
        
        # 确保labels长度与input_ids一致
        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }


class CoTSFTDataset(Dataset):
    """
    Chain-of-Thought SFT数据集
    包含思考过程（路径）和答案，对思考过程和答案都计算loss
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        system_prompt: str = "",
        instruction_template: str = "{system_prompt}\n\n{input}\n\nAnswer:",
        cot_prefix: str = " Let me think step by step:" #这里有空格是因为cot_prefix是前缀，需要与input_text拼接
    ):
        """
        初始化数据集
        
        Args:
            data_file: JSONL数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            max_samples: 最大样本数（None表示全部）
            system_prompt: 系统提示词
            instruction_template: 指令模板
            cot_prefix: CoT前缀文本
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.instruction_template = instruction_template
        self.cot_prefix = cot_prefix
        
        # 加载原始数据
        raw_data = load_jsonl(data_file, max_samples)
        
        # 转换为训练样本
        self.samples = []
        for record in raw_data:
            try:
                # 图转自然语言
                context, question = graph_to_natural_language(record)
                answer = record.get("answer", "")
                paths = record.get("paths", [])
                names = {int(k): v for k, v in record.get("names", {}).items()}
                
                if not answer or not paths:
                    continue
                
                # 选择第一条路径（如果有多条）
                path = paths[0] if paths else []
                
                # 将路径转换为CoT文本
                cot_text = path_to_cot_text(path, names, answer)
                
                # 格式化输入
                input_text = f"{context} {question}".strip()
                if system_prompt:
                    if "{system_prompt}" in instruction_template and "{input}" in instruction_template:
                        input_text = instruction_template.format(
                            system_prompt=system_prompt,
                            input=input_text
                        )
                    else:
                        input_text = f"{system_prompt}\n\n{input_text}"
                
                # 输出包含CoT和答案
                output_text = f"{self.cot_prefix} {cot_text}"
                
                self.samples.append({
                    "input": input_text,
                    "output": output_text,
                    "full_text": f"{input_text}{output_text}",
                    "path": path,
                    "answer": answer
                })
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} CoT samples from {data_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_text = sample["input"]
        output_text = sample["output"]
        
        # 分别对input和output进行编码
        input_encoded = self.tokenizer(
            input_text,
            add_special_tokens=True,
            return_tensors=None
        )
        input_ids_input = input_encoded["input_ids"]
        
        # 对output编码（不添加特殊token）
        output_encoded = self.tokenizer(
            output_text,
            add_special_tokens=False,
            return_tensors=None
        )
        output_ids = output_encoded["input_ids"]
        
        # 组合input和output
        input_ids = input_ids_input + output_ids
        
        # 截断到最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # 创建labels：input部分设为-100（不计算loss），output部分（CoT+答案）全部保留
        input_len = len(input_ids_input)
        # 对于CoT，整个output部分都计算loss
        labels = [-100] * min(input_len, len(input_ids)) + input_ids[min(input_len, len(input_ids)):]
        
        # 确保labels长度与input_ids一致
        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }

def collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer, pad_to_multiple_of: int = 8):
    """
    批处理函数
    
    Args:
        batch: 批次数据
        tokenizer: 分词器
        pad_to_multiple_of: 填充到该数的倍数
    """
    # 获取最大长度
    max_len = max(len(item["input_ids"]) for item in batch)
    
    # 填充到pad_to_multiple_of的倍数
    if pad_to_multiple_of > 0:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    
    # 获取pad_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # 填充所有序列
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        padding_len = max_len - seq_len
        
        input_ids.append(item["input_ids"] + [pad_token_id] * padding_len)
        attention_mask.append(item["attention_mask"] + [0] * padding_len)
        labels.append(item["labels"] + [-100] * padding_len)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

