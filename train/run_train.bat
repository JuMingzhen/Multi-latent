@echo off
REM SFT训练启动脚本 (Windows)

set CONFIG=train/config.json
set NUM_GPUS=1
set RESUME_CHECKPOINT=

REM 解析参数
:parse_args
if "%1"=="" goto run
if "%1"=="--config" (
    set CONFIG=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--nproc_per_node" (
    set NUM_GPUS=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--resume" (
    set RESUME_CHECKPOINT=%2
    shift
    shift
    goto parse_args
)
shift
goto parse_args

:run
if %NUM_GPUS%==1 (
    REM 单卡训练
    python train/train_sft.py --config %CONFIG%
) else (
    REM 多卡训练
    torchrun --nproc_per_node=%NUM_GPUS% train/train_sft.py --config %CONFIG%
)

if not "%RESUME_CHECKPOINT%"=="" (
    python train/train_sft.py --config %CONFIG% --resume_from_checkpoint %RESUME_CHECKPOINT%
)

