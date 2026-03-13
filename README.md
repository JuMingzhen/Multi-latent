# Multi-latent Reasoning Project

基于图结构的逻辑推理任务：数据集生成、模型训练与评测的完整框架。

## 📁 项目结构

```
.
├── data/                    # 数据文件夹
│   ├── *.jsonl            # 测试数据集（easy/middle/hard）
│   └── scripts/           # 数据生成和可视化工具
│       ├── generate_prosqa.py    # ProsQA数据集生成
│       ├── visualize_prosqa.py   # 图结构可视化
│       └── config/               # 配置文件
│
├── train/                  # 模型训练模块
│   ├── train_sft.py       # SFT训练主脚本
│   ├── dataset.py         # 数据集处理（支持多种训练策略）
│   ├── utils.py           # 工具函数
│   ├── config.json        # 训练配置文件
│   └── checkpoints/       # 训练checkpoint输出目录
│
├── eval/                   # 模型评测模块
│   ├── evaluate.py        # 评测主脚本
│   ├── config.json        # 评测配置文件
│   └── results/           # 评测结果输出目录
│
└── requirements.txt       # 统一依赖文件
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

项目已包含测试数据：
- `data/easy_test.jsonl` - 简单难度测试集
- `data/middle_test.jsonl` - 中等难度测试集
- `data/hard_test.jsonl` - 困难难度测试集

如需生成新数据，使用数据生成脚本：
```bash
cd data/scripts
pip install -r requirements_prosqa.txt
python generate_prosqa.py --num 1000 --output ../new_data.jsonl
```

### 3. 模型训练

#### 单卡训练
```bash
python train/train_sft.py --config train/config.json
```

#### 多卡分布式训练
```bash
# 2卡训练
torchrun --nproc_per_node=2 train/train_sft.py --config train/config.json

# 4卡训练
torchrun --nproc_per_node=4 train/train_sft.py --config train/config.json
```

#### 从Checkpoint恢复
```bash
python train/train_sft.py \
    --config train/config.json \
    --resume_from_checkpoint train/checkpoints/checkpoint-1000
```

### 4. 模型评测

#### 使用本地模型
```bash
python eval/evaluate.py --config eval/config.json
```

#### 使用API模型
```bash
# 设置API密钥
export OPENAI_API_KEY="your-api-key-here"
python eval/evaluate.py --config eval/config.json
```

## 📚 功能模块详解

### 1. 数据生成模块 (`data/scripts/`)

#### 生成ProsQA数据集
```bash
python data/scripts/generate_prosqa.py \
    --num 1000 \
    --output data/new_dataset.jsonl \
    --n-nodes 20
```

**功能**：
- 基于DAG（有向无环图）生成逻辑推理问题
- 支持自定义节点数、实体名称、概念名称
- 自动生成问题、答案和推理路径

#### 可视化图结构
```bash
# 可视化JSONL文件中的第0条数据
python data/scripts/visualize_prosqa.py data/easy_test.jsonl --index 0

# 保存为图片
python data/scripts/visualize_prosqa.py data/easy_test.jsonl -i 0 -o graph.png
```

**功能**：
- 可视化DAG结构
- 高亮显示推理路径
- 支持保存为PNG图片

### 2. 模型训练模块 (`train/`)

#### 支持的训练策略

**Answer-Only（只有答案）**
- 只对答案部分计算loss
- 训练速度快，适合快速迭代
- 配置：`"dataset_type": "answer_only"`

**CoT (Chain-of-Thought，思维链)**
- 包含推理过程，对思考过程和答案都计算loss
- 提高模型可解释性
- 配置：`"dataset_type": "cot"`

#### 训练配置示例

```json
{
  "model": {
    "name_or_path": "gpt2"
  },
  "data": {
    "train_file": "data/easy_test.jsonl",
    "dataset_type": "answer_only",  // 或 "cot"
    "max_seq_length": 512
  },
  "training": {
    "output_dir": "train/checkpoints",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 5e-5,
    "fp16": true
  }
}
```

#### 训练特性

- ✅ **多卡分布式训练**：支持DDP，自动处理多GPU
- ✅ **Checkpoint管理**：自动保存和恢复，支持断点续训
- ✅ **混合精度训练**：支持FP16/BF16
- ✅ **梯度检查点**：节省内存
- ✅ **图转自然语言**：自动将图结构转换为训练文本

### 3. 模型评测模块 (`eval/`)

#### 支持的评测方式

**本地模型评测**
- 支持通过transformers加载的模型（GPT2、LLaMA等）
- 自动处理GPU/CPU设备选择

**API模型评测**
- 支持OpenAI API
- 支持自定义API端点（如通义千问等）

#### 评测配置示例

```json
{
  "model": {
    "type": "local",  // 或 "api"
    "name": "gpt2"
  },
  "data": {
    "test_files": [
      "data/easy_test.jsonl",
      "data/middle_test.jsonl",
      "data/hard_test.jsonl"
    ]
  },
  "system_prompt": "You are a logical reasoning assistant.",
  "output_dir": "eval/results"
}
```

#### 评测输出

评测结果保存在 `eval/results/` 目录：
- `{test_file}_results.json` - 每个测试文件的详细结果
  - `summary`: 总体统计（准确率、样本数、耗时等）
  - `results`: 每个样本的详细结果

## 🔧 配置说明

### 训练配置 (`train/config.json`)

主要配置项：
- `model.name_or_path`: 模型名称或路径
- `data.dataset_type`: 数据集类型（`"answer_only"` 或 `"cot"`）
- `data.max_seq_length`: 最大序列长度
- `training.*`: 训练超参数（学习率、批次大小等）
- `distributed.*`: 分布式训练配置

详细配置说明见 `train/README.md`

### 评测配置 (`eval/config.json`)

主要配置项：
- `model.type`: 模型类型（`"local"` 或 `"api"`）
- `data.test_files`: 测试文件列表
- `system_prompt`: 系统提示词
- `max_tokens`: 最大生成token数

详细配置说明见 `eval/README.md`

## 📊 数据格式

### 输入数据格式（JSONL）

每行一个JSON对象：

```json
{
  "question": "Is Daniel a orpus or fompus?",
  "answer": "fompus",
  "edges": [[0, 2], [1, 3], [2, 4], ...],
  "names": {
    "0": "Daniel",
    "1": "Parker",
    "2": "korpal",
    ...
  },
  "labels": {"0": 1, "1": 2, ...},
  "paths": [[0, 2, 15, 19]],
  "concept_a_node": 19,
  "concept_b_node": 16
}
```

### 图转自然语言

系统自动将图结构转换为自然语言：

**输入**：
- 图结构（edges, names）

**输出**：
- 实体到概念：`"Daniel is a korpal."`
- 概念到概念：`"Every korpal is a sherpus."`
- 完整上下文：`"Daniel is a korpal. Every korpal is a sherpus. ... Is Daniel a orpus or fompus?"`

## 🎯 使用场景

### 场景1：快速训练和评测

```bash
# 1. 训练模型（Answer-Only，快速）
# 修改 train/config.json: "dataset_type": "answer_only"
python train/train_sft.py --config train/config.json

# 2. 评测模型
python eval/evaluate.py --config eval/config.json
```

### 场景2：训练可解释模型

```bash
# 1. 训练CoT模型
# 修改 train/config.json: "dataset_type": "cot"
python train/train_sft.py --config train/config.json

# 2. 评测模型
python eval/evaluate.py --config eval/config.json
```

### 场景3：多模型对比评测

```bash
# 评测多个模型
python eval/evaluate.py --config eval/config_example_local.json
python eval/evaluate.py --config eval/config_example_api.json
```

## 🛠️ 工具脚本

### 测试数据集

```bash
# 测试Answer-Only数据集
python train/test_datasets.py

# 测试图转自然语言
python eval/test_graph_to_nl.py
```

### 数据可视化

```bash
# 可视化图结构
python data/scripts/visualize_prosqa.py data/easy_test.jsonl --index 0 -o graph.png
```

## 📝 常见问题

### 训练相关问题

**Q: 内存不足怎么办？**
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing`
- 使用 `fp16` 或 `bf16`

**Q: 多卡训练速度慢？**
- 增加 `dataloader_num_workers`
- 检查GPU间通信
- 设置 `find_unused_parameters: false`

**Q: 如何选择数据集类型？**
- `answer_only`: 快速训练，直接学习答案
- `cot`: 学习推理过程，提高可解释性

### 评测相关问题

**Q: API调用失败？**
- 检查API密钥是否正确
- 检查网络连接
- 检查API配额和限制

**Q: 本地模型加载失败？**
- 确保有足够的GPU/CPU内存
- 检查模型路径是否正确
- 尝试使用CPU：`"device": "cpu"`

### 数据相关问题

**Q: 数据格式不正确？**
- 确保包含必要字段：`question`, `answer`, `edges`, `names`
- CoT训练需要：`paths` 字段
- 检查JSONL格式是否正确

## 📖 详细文档

- **训练模块**：详见 `train/README.md`
- **评测模块**：详见 `eval/README.md`
- **数据生成**：详见 `data/scripts/` 中的脚本注释

## 🔬 实验建议

### 实验1：对比不同训练策略

```bash
# 训练Answer-Only模型
python train/train_sft.py --config train/config_answer_only.json

# 训练CoT模型
python train/train_sft.py --config train/config_cot.json

# 分别评测
python eval/evaluate.py --config eval/config.json
```

### 实验2：不同难度数据集

```bash
# 在easy/middle/hard数据集上分别训练和评测
# 修改配置文件中的 train_file 和 test_files
```

### 实验3：不同模型架构

```bash
# 修改 config.json 中的 model.name_or_path
# 尝试不同的预训练模型（gpt2, gpt2-medium, gpt2-large等）
```

## 📄 许可证

本项目用于研究目的。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请提交Issue或联系项目维护者。

