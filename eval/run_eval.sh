#!/bin/bash
# 评测脚本，支持单卡和多卡分布式评测

# 设置默认值
CONFIG="eval/config.json"
NUM_GPUS=1
OUTPUT_DIR=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --nproc_per_node)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 构建命令
if [ $NUM_GPUS -eq 1 ]; then
    # 单卡评测
    CMD="python eval/evaluate.py --config $CONFIG"
else
    # 多卡分布式评测
    CMD="torchrun --nproc_per_node=$NUM_GPUS eval/evaluate.py --config $CONFIG"
fi

# 添加output参数
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

# 执行命令
echo "Running: $CMD"
eval $CMD

