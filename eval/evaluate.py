#!/usr/bin/env python3
"""
大模型能力评测脚本
支持本地模型和API调用两种策略
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有 tqdm，创建一个简单的占位符
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import torch
    import torch.distributed as dist
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
    """本地模型调用接口，支持本地路径和HuggingFace模型名称"""
    
    def __init__(self, model_name: str, device: str = "cuda", trust_remote_code: bool = False, 
                 local_rank: int = -1, torch_dtype: Optional[str] = None):
        """
        初始化本地模型
        
        Args:
            model_name: 模型名称或本地路径
            device: 设备类型 ("cuda" 或 "cpu")
            trust_remote_code: 是否信任远程代码
            local_rank: 本地GPU rank（用于分布式，-1表示不使用分布式）
            torch_dtype: 模型数据类型 ("float16", "bfloat16", "float32")
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for local models. Install with: pip install transformers torch")
        
        # 确定设备
        if local_rank >= 0:
            # 分布式模式，使用指定的local_rank对应的GPU
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 确定数据类型
        if torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # 检查是本地路径还是HuggingFace模型名称
        model_path = Path(model_name)
        is_local_path = model_path.exists() and model_path.is_dir()
        
        rank = int(os.environ.get("RANK", -1))
        if rank >= 0:
            print(f"[Rank {rank}] Loading model from {'local path' if is_local_path else 'HuggingFace'}: {model_name} on {self.device}...")
        else:
            print(f"Loading model from {'local path' if is_local_path else 'HuggingFace'}: {model_name} on {self.device}...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype if dtype != torch.float32 else None
        )
        
        # 移动到指定设备
        if local_rank >= 0:
            self.model.to(self.device)
        else:
            self.model.to(self.device)
        
        self.model.eval()
        
        if rank >= 0:
            print(f"[Rank {rank}] Model loaded successfully.")
        else:
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


def extract_answer(response: str, options: List[str], entity: Optional[str] = None) -> Optional[str]:
    """
    从模型响应中提取答案
    优先使用固定结构 "{entity} is a {concept}" 提取
    
    Args:
        response: 模型响应文本
        options: 候选答案列表
        entity: 实体名称（用于匹配固定结构）
    
    Returns:
        匹配到的答案，如果没有匹配则返回None
    """
    if not options:
        return None
    
    response_lower = response.lower().strip()
    options_lower = [opt.lower() for opt in options]
    
    # 优先方法：如果提供了entity，尝试从固定结构 "{entity} is a {concept}" 中提取
    if entity:
        entity_lower = entity.lower()
        # 匹配模式：{entity} is a {concept}
        # 使用正则表达式匹配，允许大小写不敏感，允许标点符号
        pattern = re.compile(
            rf'\b{re.escape(entity_lower)}\s+is\s+a\s+(\w+)',
            re.IGNORECASE
        )
        matches = pattern.findall(response_lower)
        
        if matches:
            # 取最后一个匹配（如果有多个）
            extracted_concept = matches[-1].strip().rstrip(".,!?;:")
            # 匹配选项
            for i, opt_lower in enumerate(options_lower):
                if opt_lower == extracted_concept:
                    return options[i]
        
        # 也尝试匹配 "is a {concept}" 结构（entity可能在前面）
        pattern2 = re.compile(
            rf'\bis\s+a\s+(\w+)',
            re.IGNORECASE
        )
        matches2 = pattern2.findall(response_lower)
        if matches2:
            # 取最后一个匹配
            extracted_concept = matches2[-1].strip().rstrip(".,!?;:")
            for i, opt_lower in enumerate(options_lower):
                if opt_lower == extracted_concept:
                    return options[i]
    
    # 回退方法1: 直接匹配（包含关系）
    for i, opt_lower in enumerate(options_lower):
        if opt_lower in response_lower:
            return options[i]
    
    # 回退方法2: 提取第一个单词并匹配
    words = response_lower.split()
    if words:
        first_word = words[0].rstrip(".,!?;:")
        for i, opt_lower in enumerate(options_lower):
            if opt_lower == first_word:
                return options[i]
    
    # 回退方法3: 提取最后一个单词并匹配（有时答案在最后）
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
    
    # 获取entity名称（通常是节点0）
    names = {int(k): v for k, v in record.get("names", {}).items()}
    entity = names.get(0, "")  # 节点0通常是实体
    
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
        concept_a_node = record.get("concept_a_node")
        concept_b_node = record.get("concept_b_node")
        if concept_a_node is not None:
            options.append(names.get(concept_a_node, ""))
        if concept_b_node is not None:
            options.append(names.get(concept_b_node, ""))
    
    predicted = extract_answer(response, options, entity)
    is_correct = predicted is not None and predicted.lower() == ground_truth.lower()
    
    return prompt, response, is_correct


def evaluate_dataset(
    model,
    test_file: str,
    config: Dict,
    output_file: Optional[str] = None,
    rank: int = -1,
    world_size: int = 1
) -> Dict:
    """
    评测整个数据集，支持分布式评测
    
    Args:
        model: 模型实例
        test_file: 测试文件路径
        config: 配置字典
        output_file: 输出文件路径
        rank: 当前进程rank（-1表示非分布式）
        world_size: 总进程数
    """
    is_distributed = rank >= 0 and world_size > 1
    
    if is_distributed:
        print(f"\n[Rank {rank}] Evaluating {test_file}...")
    else:
        print(f"\nEvaluating {test_file}...")
    
    # 加载所有测试数据
    all_test_data = load_test_data(test_file, config["data"].get("max_samples"))
    
    # 分布式模式下，每个进程处理一部分数据
    start_idx = 0
    if is_distributed:
        # 将数据分片
        chunk_size = len(all_test_data) // world_size
        start_idx = rank * chunk_size
        if rank == world_size - 1:
            # 最后一个进程处理剩余的所有数据
            end_idx = len(all_test_data)
        else:
            end_idx = start_idx + chunk_size
        test_data = all_test_data[start_idx:end_idx]
        print(f"[Rank {rank}] Processing samples {start_idx} to {end_idx-1} (total: {len(test_data)} samples)")
    else:
        test_data = all_test_data
        print(f"Loaded {len(test_data)} samples")
    
    is_local = config["model"]["type"] == "local"
    system_prompt = config.get("system_prompt", "")
    max_tokens = config.get("max_tokens", 50)
    temperature = config.get("temperature", 0.0)
    
    results = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    # 只在主进程（rank 0）或非分布式模式下显示进度条
    show_progress = (not is_distributed or rank == 0)
    
    # 创建进度条
    if show_progress:
        pbar = tqdm(
            enumerate(test_data),
            total=len(test_data),
            desc=f"Evaluating {Path(test_file).stem}",
            unit="sample",
            ncols=100
        )
    else:
        pbar = enumerate(test_data)
    
    for i, record in pbar:
        try:
            prompt, response, is_correct = evaluate_single(
                model, record, system_prompt, is_local, max_tokens, temperature
            )
            
            # 在分布式模式下，需要保存原始索引
            original_idx = start_idx + i if is_distributed else i
            
            results.append({
                "index": original_idx,
                "question": record.get("question", ""),
                "ground_truth": record["answer"],
                "predicted": response,
                "is_correct": is_correct,
                "prompt": prompt
            })
            
            if is_correct:
                correct += 1
            total += 1
            
            # 更新进度条显示准确率
            if show_progress and HAS_TQDM:
                current_accuracy = correct / total if total > 0 else 0.0
                pbar.set_postfix({
                    "accuracy": f"{current_accuracy:.4f}",
                    "correct": f"{correct}/{total}"
                })
        
        except Exception as e:
            original_idx = start_idx + i if is_distributed else i
            error_msg = f"Error processing sample {original_idx}: {e}"
            if is_distributed and not show_progress:
                print(f"[Rank {rank}] {error_msg}")
            elif not is_distributed:
                print(error_msg)
            
            results.append({
                "index": original_idx,
                "question": record.get("question", ""),
                "ground_truth": record["answer"],
                "predicted": "ERROR",
                "is_correct": False,
                "error": str(e)
            })
            total += 1
    
    # 关闭进度条
    if show_progress and HAS_TQDM:
        pbar.close()
    
    elapsed_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0
    
    # 分布式模式下，收集所有进程的结果
    if is_distributed and HAS_TRANSFORMERS:
        # 收集所有进程的结果
        all_results = [None] * world_size
        all_correct = [0] * world_size
        all_total = [0] * world_size
        
        dist.all_gather_object(all_results, results)
        dist.all_gather_object(all_correct, correct)
        dist.all_gather_object(all_total, total)
        
        # 只在主进程（rank 0）合并结果
        if rank == 0:
            # 合并所有结果
            all_results_flat = []
            for r in all_results:
                all_results_flat.extend(r)
            
            # 按索引排序
            all_results_flat.sort(key=lambda x: x["index"])
            
            # 计算总体统计
            total_correct = sum(all_correct)
            total_samples = sum(all_total)
            total_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            
            results = all_results_flat
            correct = total_correct
            total = total_samples
            accuracy = total_accuracy
        else:
            # 非主进程返回空结果
            return {
                "test_file": test_file,
                "total_samples": 0,
                "correct": 0,
                "accuracy": 0.0,
                "elapsed_time": elapsed_time,
                "samples_per_second": 0
            }
    
    summary = {
        "test_file": test_file,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "elapsed_time": elapsed_time,
        "samples_per_second": total / elapsed_time if elapsed_time > 0 else 0
    }
    
    if is_distributed:
        print(f"\n[Rank {rank}] Results for {test_file}:")
    else:
        print(f"\nResults for {test_file}:")
    if not is_distributed or rank == 0:
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Speed: {summary['samples_per_second']:.2f} samples/s")
    
    # 只在主进程（或非分布式模式）保存结果
    if output_file and (not is_distributed or rank == 0):
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


def setup_distributed():
    """初始化分布式评测"""
    if not HAS_TRANSFORMERS:
        return -1, 1, -1
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # 初始化分布式进程组
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        return rank, world_size, local_rank
    return -1, 1, -1


def cleanup_distributed():
    """清理分布式评测"""
    if HAS_TRANSFORMERS and dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="评测大模型能力")
    parser.add_argument("--config", type=str, default="eval/config.json", help="配置文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出目录（默认使用配置文件中的output_dir）")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地GPU rank（用于分布式评测）")
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    is_distributed = rank >= 0 and world_size > 1
    
    # 加载配置
    config = load_config(args.config)
    if is_distributed:
        print(f"[Rank {rank}] Loaded config from {args.config}")
    else:
        print(f"Loaded config from {args.config}")
    
    # 初始化模型
    if config["model"]["type"] == "local":
        if is_distributed:
            print(f"[Rank {rank}] Using local model...")
        else:
            print("Using local model...")
        
        model = LocalModel(
            model_name=config["model"]["name"],
            device=config["model"].get("device", "cuda"),
            trust_remote_code=config["model"].get("trust_remote_code", False),
            local_rank=local_rank,
            torch_dtype=config["model"].get("torch_dtype", None)
        )
        is_local = True
    else:
        if is_distributed:
            print(f"[Rank {rank}] Using API model...")
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
    if not is_distributed or rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 等待所有进程
    if is_distributed and HAS_TRANSFORMERS:
        dist.barrier()
    
    # 评测所有测试文件
    all_summaries = []
    test_files = config["data"]["test_files"]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            if is_distributed:
                print(f"[Rank {rank}] Warning: Test file {test_file} not found, skipping...")
            else:
                print(f"Warning: Test file {test_file} not found, skipping...")
            continue
        
        # 生成输出文件名
        test_name = Path(test_file).stem
        model_name = config['api']['model'] if config['model']['type'] == 'api' else config['model']['name']
        # 清理模型名称中的路径分隔符
        model_name_clean = model_name.replace("/", "_").replace("\\", "_")
        output_file = os.path.join(output_dir, f"{test_name}_{model_name_clean}_results.json")
        
        summary = evaluate_dataset(model, test_file, config, output_file, rank, world_size)
        if summary["total_samples"] > 0:  # 只添加有效的结果
            all_summaries.append(summary)
    
    # 打印总体统计（只在主进程或非分布式模式）
    if (not is_distributed or rank == 0) and len(all_summaries) > 1:
        print("\n" + "="*50)
        print("Overall Summary:")
        print("="*50)
        for summary in all_summaries:
            print(f"  {Path(summary['test_file']).stem}: {summary['accuracy']:.4f}")
    
    # 清理分布式
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()

