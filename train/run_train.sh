#!/bin/bash
# SFT训练启动脚本

# 设置默认值
CONFIG="train/config.json"
NUM_GPUS=1
RESUME_CHECKPOINT=""

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
        --resume)
            RESUME_CHECKPOINT="$2"
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
    # 单卡训练
    CMD="python train/train_sft.py --config $CONFIG"
else
    # 多卡训练
    CMD="torchrun --nproc_per_node=$NUM_GPUS train/train_sft.py --config $CONFIG"
fi

# 添加resume参数
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
fi

# 执行命令
echo "Running: $CMD"
eval $CMD

