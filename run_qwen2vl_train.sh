#!/bin/bash
# LoRA SFT + FFN suppression at layer 12 for Qwen2.5-VL-7B-Instruct
# Run: bash run_qwen2vl_train.sh 2>&1 | tee run_qwen2vl_train.log

MODEL=/home/aied_test/models/Qwen2.5-VL-7B-Instruct
TRAIN_FILE=./data/train/pip_kag_train.jsonl
OUTPUT_DIR=./checkpoints/qwen2vl_lora_layer12_lambda0.5

echo "========================================"
echo "LoRA SFT + suppression at layer 12 (lambda=0.5)"
echo "========================================"

CUDA_VISIBLE_DEVICES=2 python src/2_tuning/train_qwen2vl.py \
    --model_name_or_path $MODEL \
    --inhibit_strength 0.5 \
    --inhibit_layer_list 12 \
    --lora_r 64 \
    --lora_alpha 64 \
    --train_file $TRAIN_FILE \
    --max_len 1024 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --gradient_checkpointing True \
    --save_steps 300 \
    --logging_steps 10 \
    --report_to none \
    --output_dir $OUTPUT_DIR

echo "========================================"
echo "Training complete. Checkpoint at $OUTPUT_DIR"
echo "========================================"
