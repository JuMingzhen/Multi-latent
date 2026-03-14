#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) 训练脚本
支持多卡分布式训练和checkpoint管理
"""

import argparse
import json
import os
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Optional
import logging
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import set_seed

from train.dataset import AnswerOnlySFTDataset, CoTSFTDataset, SFTDataset
from train.utils import load_jsonl


def setup_logging(rank: int = 0):
    """设置日志系统，只在主进程输出详细日志"""
    if rank == 0:
        log_filename = f"train_sft_{rank}.log"
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_filename),  # 写入文件
                logging.StreamHandler()             # 同时打印控制台
            ]
        )
    else:
        # 非主进程只显示WARNING及以上级别，减少输出
        logging.basicConfig(
            format='%(asctime)s - [RANK-%d] - %(levelname)s - %(message)s' % rank,
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.WARNING
        )
    
    return logging.getLogger(__name__)


def setup_distributed():
    """初始化分布式训练"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def get_model_and_tokenizer(config: Dict, local_rank: int = 0, logger=None):
    """加载模型和分词器"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model_config = config["model"]
    
    # 加载分词器
    logger.info(f"Loading tokenizer from {model_config['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False)
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    logger.info(f"Loading model from {model_config['name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
        torch_dtype=torch.float16 if config["training"].get("fp16", False) else torch.float32,
        device_map={"": f"cuda:{local_rank}"} if torch.cuda.is_available() else None
    )
    
    # 启用梯度检查点（如果配置）
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    return model, tokenizer


def create_data_collator(tokenizer):
    """创建数据整理器"""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )


def main():
    parser = argparse.ArgumentParser(description="SFT训练脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从checkpoint恢复训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地GPU rank（用于分布式训练）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    rank = int(os.environ.get("RANK", -1))
    logger = setup_logging(rank)
    logger.info(f"Loaded config from {args.config}")
    
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    # 设置随机种子
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # 加载模型和分词器
    model, tokenizer = get_model_and_tokenizer(config, local_rank, logger)
    
    # 准备数据集
    data_config = config["data"]
    dataset_type = data_config.get("dataset_type", "answer_only")  # 默认使用answer_only
    
    # 根据类型选择数据集类
    if dataset_type == "cot" or dataset_type == "chain_of_thought":
        DatasetClass = CoTSFTDataset
        logger.info("Using CoT (Chain-of-Thought) dataset")
    else:
        DatasetClass = AnswerOnlySFTDataset
        logger.info("Using Answer-Only dataset")
    
    # 创建训练数据集
    dataset_kwargs = {
        "data_file": data_config["train_file"],
        "tokenizer": tokenizer,
        "max_length": data_config.get("max_seq_length", 512),
        "max_samples": data_config.get("max_samples"),
        "system_prompt": config["prompt_template"].get("system_prompt", ""),
        "instruction_template": config["prompt_template"].get("instruction_template", "{system_prompt}\n\n{input}\n\nAnswer:")
    }
    
    # CoT数据集可能有额外参数
    if DatasetClass == CoTSFTDataset:
        dataset_kwargs["cot_prefix"] = data_config.get("cot_prefix", "Let me think step by step:")
    
    train_dataset = DatasetClass(**dataset_kwargs)
    
    # 验证集（如果有）
    eval_dataset = None
    if data_config.get("val_file"):
        eval_dataset = DatasetClass(
            data_file=data_config["val_file"],
            tokenizer=tokenizer,
            max_length=data_config.get("max_seq_length", 512),
            max_samples=None,
            system_prompt=config["prompt_template"].get("system_prompt", ""),
            instruction_template=config["prompt_template"].get("instruction_template", "{system_prompt}\n\n{input}\n\nAnswer:"),
            **({"cot_prefix": data_config.get("cot_prefix", "Let me think step by step:")} if DatasetClass == CoTSFTDataset else {})
        )
    
    # 数据整理器
    data_collator = create_data_collator(tokenizer)
    
    # 训练参数
    training_config = config["training"]
    output_dir = training_config["output_dir"]
    
    # 如果是分布式训练，只在主进程创建输出目录
    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 等待所有进程
    if dist.is_initialized():
        dist.barrier()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        learning_rate=training_config.get("learning_rate", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        warmup_steps=training_config.get("warmup_steps"),
        warmup_ratio=training_config.get("warmup_ratio"),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 500),
        eval_steps=training_config.get("eval_steps"),
        save_total_limit=training_config.get("save_total_limit", 3),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        remove_unused_columns=training_config.get("remove_unused_columns", False),
        ddp_find_unused_parameters=config["distributed"].get("find_unused_parameters", False),
        local_rank=local_rank,
        save_strategy="steps" if training_config.get("save_steps") else "epoch",
        evaluation_strategy="steps" if eval_dataset and training_config.get("eval_steps") else "no",
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=config["training"].get("report_to", "none"),       # 开启wandb
        run_name=config["training"].get("run_name", "my-gpt2-run"),
        project=config["training"].get("project", "llm-training")
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 从checkpoint恢复（如果指定）
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    
    # 开始训练
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 保存最终模型
    if rank == 0:
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # 保存训练指标
        metrics = train_result.metrics
        metrics_file = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Training metrics saved to {metrics_file}")
    
    # 清理
    cleanup_distributed()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

