#!/bin/bash
set -e

INPUT_DATA="./data/func_data/draw_acctivations_shuf1k.jsonl"
OUTPUT_DIR="./data/func_data/activations"
SCRIPT="./src/1_Identifying/1_collect.py"

mkdir -p ${OUTPUT_DIR}

echo "============================================="
echo "  Phase 1: Activation Collection (all models)"
echo "============================================="

echo ""
echo ">>> [1/5] LLaMA-3.1-8B-Instruct"
CUDA_VISIBLE_DEVICES=0 python3 ${SCRIPT} \
    --in_file_path ${INPUT_DATA} \
    --output_path ${OUTPUT_DIR}/activations_llama31_8b.jsonl \
    --pretrained_model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model_type llama31

echo ""
echo ">>> [2/5] Qwen2.5-VL-7B-Instruct"
CUDA_VISIBLE_DEVICES=0 python3 ${SCRIPT} \
    --in_file_path ${INPUT_DATA} \
    --output_path ${OUTPUT_DIR}/activations_qwen25vl_7b.jsonl \
    --pretrained_model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen25vl

echo ""
echo ">>> [3/5] Gemma-3-12B-it"
CUDA_VISIBLE_DEVICES=0 python3 ${SCRIPT} \
    --in_file_path ${INPUT_DATA} \
    --output_path ${OUTPUT_DIR}/activations_gemma3_12b.jsonl \
    --pretrained_model_path google/gemma-3-12b-it \
    --model_type gemma3

echo ""
echo ">>> [4/5] Mistral-7B-Instruct-v0.3"
CUDA_VISIBLE_DEVICES=0 python3 ${SCRIPT} \
    --in_file_path ${INPUT_DATA} \
    --output_path ${OUTPUT_DIR}/activations_mistral_7b.jsonl \
    --pretrained_model_path mistralai/Mistral-7B-Instruct-v0.3 \
    --model_type mistral

echo ""
echo ">>> [5/5] LLaMA-3.1-70B-Instruct"
CUDA_VISIBLE_DEVICES=0,2 python3 ${SCRIPT} \
    --in_file_path ${INPUT_DATA} \
    --output_path ${OUTPUT_DIR}/activations_llama31_70b.jsonl \
    --pretrained_model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
    --model_type llama31

echo ""
echo "============================================="
echo "  All done! Activation files saved to:"
echo "  ${OUTPUT_DIR}/"
echo "============================================="
ls -lh ${OUTPUT_DIR}/
