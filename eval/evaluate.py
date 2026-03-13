#!/usr/bin/env python3
"""
大模型能力评测脚本
支持本地模型和API调用两种策略
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def graph_to_natural_language(record: Dict) -> str:
    """
    将图结构转换为自然语言描述
    
    规则：
    - 对于概念到概念的关系：[u, v] -> "Every {names[u]} is a {names[v]}."
    - 对于实体到概念的关系：[u, v] where u is entity -> "{names[u]} is a {names[v]}."
    
    实体识别：
    - 节点0和1通常是实体（根节点）
    - 如果labels存在，label为0的节点也可能是实体（未连接到根节点的节点）
    - 实体节点通常是人名（首字母大写）
    """
    edges = record["edges"]
    names = {int(k): v for k, v in record["names"].items()}
    
    # 识别实体节点
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
    
    # 添加问题
    question = record.get("question", "")
    if question:
        full_prompt = f"{context} {question}"
    else:
        full_prompt = context
    
    return full_prompt


class LocalModel:
    """本地模型调用接口"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for local models. Install with: pip install transformers torch")
        
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully.")
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 只返回新生成的部分
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()


class APIModel:
    """API调用接口"""
    
    def __init__(self, provider: str, model: str, api_key: str = "", base_url: str = ""):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        
        if self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai library is required. Install with: pip install openai")
            if not self.api_key:
                raise ValueError("API key is required for OpenAI. Set it in config or OPENAI_API_KEY environment variable.")
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url if self.base_url else None)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 50, temperature: float = 0.0) -> str:
        """生成回答"""
        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            time.sleep(0.5)
            
            return response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")


def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """加载测试数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_answer(response: str, options: List[str]) -> Optional[str]:
    """
    从模型响应中提取答案
    尝试匹配选项中的任何一个
    """
    if not options:
        return None
    
    response_lower = response.lower().strip()
    options_lower = [opt.lower() for opt in options]
    
    # 方法1: 直接匹配（包含关系）
    for i, opt_lower in enumerate(options_lower):
        if opt_lower in response_lower:
            return options[i]
    
    # 方法2: 提取第一个单词并匹配
    words = response_lower.split()
    if words:
        first_word = words[0].rstrip(".,!?;:")
        for i, opt_lower in enumerate(options_lower):
            if opt_lower == first_word:
                return options[i]
    
    # 方法3: 提取最后一个单词并匹配（有时答案在最后）
    if words:
        last_word = words[-1].rstrip(".,!?;:")
        for i, opt_lower in enumerate(options_lower):
            if opt_lower == last_word:
                return options[i]
    
    return None


def evaluate_single(
    model, 
    record: Dict, 
    system_prompt: str = "",
    is_local: bool = True,
    max_tokens: int = 50,
    temperature: float = 0.0
) -> Tuple[str, str, bool]:
    """
    评测单个样本
    
    返回: (prompt, response, is_correct)
    """
    # 转换为自然语言
    prompt = graph_to_natural_language(record)
    
    # 生成回答
    if is_local:
        response = model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
    else:
        response = model.generate(prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature)
    
    # 提取答案
    ground_truth = record["answer"]
    # 从问题中提取选项
    question = record.get("question", "")
    options = []
    
    # 方法1: 从问题中提取（格式：Is X a Y or Z?）
    if " or " in question.lower():
        parts = question.lower().split(" or ")
        if len(parts) == 2:
            # 提取第一个选项（在"or"之前）
            opt1_part = parts[0]
            # 找到"a "之后的内容
            if " a " in opt1_part:
                opt1 = opt1_part.split(" a ")[-1].strip().rstrip("?")
            else:
                opt1 = opt1_part.split()[-1].rstrip("?")
            
            # 提取第二个选项（在"or"之后）
            opt2 = parts[1].strip().rstrip("?")
            
            options = [opt1, opt2]
    
    # 方法2: 如果没有从问题中提取到，使用concept_a_node和concept_b_node
    if not options:
        names = {int(k): v for k, v in record["names"].items()}
        concept_a_node = record.get("concept_a_node")
        concept_b_node = record.get("concept_b_node")
        if concept_a_node is not None:
            options.append(names.get(concept_a_node, ""))
        if concept_b_node is not None:
            options.append(names.get(concept_b_node, ""))
    
    predicted = extract_answer(response, options)
    is_correct = predicted is not None and predicted.lower() == ground_truth.lower()
    
    return prompt, response, is_correct


def evaluate_dataset(
    model,
    test_file: str,
    config: Dict,
    output_file: Optional[str] = None
) -> Dict:
    """评测整个数据集"""
    print(f"\nEvaluating {test_file}...")
    
    test_data = load_test_data(test_file, config["data"].get("max_samples"))
    print(f"Loaded {len(test_data)} samples")
    
    is_local = config["model"]["type"] == "local"
    system_prompt = config.get("system_prompt", "")
    max_tokens = config.get("max_tokens", 50)
    temperature = config.get("temperature", 0.0)
    
    results = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for i, record in enumerate(test_data):
        try:
            prompt, response, is_correct = evaluate_single(
                model, record, system_prompt, is_local, max_tokens, temperature
            )
            
            results.append({
                "index": i,
                "question": record.get("question", ""),
                "ground_truth": record["answer"],
                "predicted": response,
                "is_correct": is_correct,
                "prompt": prompt
            })
            
            if is_correct:
                correct += 1
            total += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_data)} samples, Accuracy: {correct/total:.4f}")
        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append({
                "index": i,
                "question": record.get("question", ""),
                "ground_truth": record["answer"],
                "predicted": "ERROR",
                "is_correct": False,
                "error": str(e)
            })
            total += 1
    
    elapsed_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0
    
    summary = {
        "test_file": test_file,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed_time,
        "samples_per_second": total / elapsed_time if elapsed_time > 0 else 0
    }
    
    print(f"\nResults for {test_file}:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Speed: {summary['samples_per_second']:.2f} samples/s")
    
    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "config": {"system_prompt": system_prompt, "temperature": temperature, "max_tokens": max_tokens},
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="评测大模型能力")
    parser.add_argument("--config", type=str, default="eval/config.json", help="配置文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录（默认使用配置文件中的output_dir）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # 初始化模型
    if config["model"]["type"] == "local":
        print("Using local model...")
        model = LocalModel(
            model_name=config["model"]["name"],
            device=config["model"].get("device", "cuda")
        )
        is_local = True
    else:
        print("Using API model...")
        api_config = config["api"]
        model = APIModel(
            provider=api_config["provider"],
            model=api_config["model"],
            api_key=api_config.get("api_key", ""),
            base_url=api_config.get("base_url", "")
        )
        is_local = False
    
    # 确定输出目录
    output_dir = args.output or config.get("output_dir", "eval/results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 评测所有测试文件
    all_summaries = []
    test_files = config["data"]["test_files"]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"Warning: Test file {test_file} not found, skipping...")
            continue
        
        # 生成输出文件名
        test_name = Path(test_file).stem
        model_name = config['api']['model'] if config['model']['type'] == 'api' else config['model']['name']
        output_file = os.path.join(output_dir, f"{test_name}_{model_name}_results.json")
        
        summary = evaluate_dataset(model, test_file, config, output_file)
        all_summaries.append(summary)
    
    # 打印总体统计
    if len(all_summaries) > 1:
        print("\n" + "="*50)
        print("Overall Summary:")
        print("="*50)
        for summary in all_summaries:
            print(f"  {Path(summary['test_file']).stem}: {summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()

