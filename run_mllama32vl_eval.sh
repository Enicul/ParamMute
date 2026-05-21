#!/bin/bash
set -euo pipefail

MODEL=/home/aied_test/models/Llama-3.2-11B-Vision-Instruct
DATA_DIR=./data/CoConflictQA
SCRIPT=src/3_evaluate/eval_CoConflictQA_mllama.py
LAYERS=(2 4 6)

DATASETS=(
    NaturalQuestionsShort_kc.jsonl
    NewsQA_kc.jsonl
    hotpotq_kc.jsonl
    SearchQA_kc.jsonl
    SQuAD_kc.jsonl
    TriviaQA-web_kc.jsonl
)

echo "========================================"
echo "Llama-3.2-11B-Vision baseline"
echo "========================================"
for DS in "${DATASETS[@]}"; do
    NAME="${DS%.jsonl}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$SCRIPT" \
        --model_name "$MODEL" \
        --data_path "$DATA_DIR/$DS" \
        --act_inhibit_ratio 1.0 \
        --act_inhibit_layer_list "${LAYERS[@]}" \
        --schema instr \
        --output_path "./results/eval/mllama32vl_baseline/$DS" \
        --log_path "./results/eval/mllama32vl_baseline/${NAME}.log"
done

echo "========================================"
echo "Llama-3.2-11B-Vision suppressed lambda=0.5 layers=${LAYERS[*]}"
echo "========================================"
for DS in "${DATASETS[@]}"; do
    NAME="${DS%.jsonl}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$SCRIPT" \
        --model_name "$MODEL" \
        --data_path "$DATA_DIR/$DS" \
        --act_inhibit_ratio 0.5 \
        --act_inhibit_layer_list "${LAYERS[@]}" \
        --schema instr \
        --output_path "./results/eval/mllama32vl_layer2_4_6_lambda0.5/$DS" \
        --log_path "./results/eval/mllama32vl_layer2_4_6_lambda0.5/${NAME}.log"
done

echo "Done. Results in ./results/eval/mllama32vl_*"
