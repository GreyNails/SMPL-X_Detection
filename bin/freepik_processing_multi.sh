#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="1"
export PYTHONUNBUFFERED="1"

# 切换到工作目录
cd "/home/dell/DataTool-HumanCentric/detection_aios/AiOS" || {
    echo "无法切换到工作目录，请检查路径是否正确"
    exit 1
}

# 执行Python程序 - 多线程版本
"/home/dell/anaconda3/envs/pymafx/bin/python" main_multi.py \
    -c "config/aios_smplx_demo.py" \
    --options \
    "batch_size=1" \
    "backbone=resnet50" \
    "threshold=0.1" \
    --resume "data/checkpoint/aios_checkpoint.pth" \
    --eval \
    --inference \
    --process_json \
    --json_dataset "/storage/HCL_data/Freepik/psd_processed/all_psd_data_detection.json" \
    --output_dir "demo/json_output_freekip" \
    --device "cuda:1"\
    --num_threads 8 \
    --batch_inference 4

# 参数说明:
# --num_threads 8: 使用8个线程并行处理数据预处理（I/O操作）
# --batch_inference 4: 每批处理4个样本
# 
# 性能调优建议:
# - num_threads 建议设置为 CPU 核心数的 1-2 倍（例如 8 核 CPU 设置为 8-16）
# - batch_inference 建议根据 GPU 内存调整：
#   * 小模型/大显存: 可以设置为 4-8
#   * 大模型/小显存: 设置为 1-2
# - 如果遇到 GPU 内存不足，减小 batch_inference
# - 如果遇到 I/O 瓶颈，增大 num_threads