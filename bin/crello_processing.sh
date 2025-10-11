#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0"
export PYTHONUNBUFFERED="1"

# 切换到工作目录
cd "/home/dell/DataTool-HumanCentric/detection_aios/AiOS" || {
    echo "无法切换到工作目录，请检查路径是否正确"
    exit 1
}

# 执行Python程序
"/home/dell/anaconda3/envs/pymafx/bin/python" main.py \
    -c "config/aios_smplx_demo.py" \
    --options \
    "batch_size=1" \
    "backbone=resnet50" \
    "threshold=0.1" \
    --resume "data/checkpoint/aios_checkpoint.pth" \
    --eval \
    --inference \
    --process_json \
    --json_dataset "/storage/HCL_data/crello/detections/dataset.json" \
    --output_dir "demo/json_output_v1"