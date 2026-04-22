#!/bin/bash
# Evaluate Qwen2.5-VL-7B-Instruct on all 6 CoConflictQA datasets
# Baseline (no suppression) + Suppressed (layer 12, lambda=0.5)
# Run: bash run_qwen2vl_eval.sh 2>&1 | tee run_qwen2vl_eval.log

MODEL=/home/aied_test/models/Qwen2.5-VL-7B-Instruct
DATA_DIR=./data/CoConflictQA
SCRIPT=src/3_evaluate/eval_CoConflictQA_qwen2vl.py

DATASETS=(
    NaturalQuestionsShort_kc.jsonl
    NewsQA_kc.jsonl
    hotpotq_kc.jsonl
    SearchQA_kc.jsonl
    SQuAD_kc.jsonl
    TriviaQA-web_kc.jsonl
)

# ── Baseline: no suppression (lambda=1.0) ────────────────────────────────────
echo "========================================"
echo "BASELINE (no suppression, lambda=1.0)"
echo "========================================"

for DS in "${DATASETS[@]}"; do
    NAME="${DS%.jsonl}"
    echo "--- Baseline: $NAME ---"
    CUDA_VISIBLE_DEVICES=2 python $SCRIPT \
        --model_name $MODEL \
        --data_path $DATA_DIR/$DS \
        --act_inhibit_ratio 1.0 \
        --act_inhibit_layer_list 12 \
        --schema instr \
        --output_path ./results/eval/qwen2vl_baseline/$DS \
        --log_path ./results/eval/qwen2vl_baseline/${NAME}.log
    echo "Done: $NAME"
done

# ── Suppressed: layer 12, lambda=0.5 ─────────────────────────────────────────
echo "========================================"
echo "SUPPRESSED (layer 12, lambda=0.5)"
echo "========================================"

for DS in "${DATASETS[@]}"; do
    NAME="${DS%.jsonl}"
    echo "--- Suppressed: $NAME ---"
    CUDA_VISIBLE_DEVICES=2 python $SCRIPT \
        --model_name $MODEL \
        --data_path $DATA_DIR/$DS \
        --act_inhibit_ratio 0.5 \
        --act_inhibit_layer_list 12 \
        --schema instr \
        --output_path ./results/eval/qwen2vl_layer12_lambda0.5/$DS \
        --log_path ./results/eval/qwen2vl_layer12_lambda0.5/${NAME}.log
    echo "Done: $NAME"
done

echo "========================================"
echo "All evaluations complete."
echo "Results in ./results/eval/"
echo "========================================"
