#!/bin/bash
MODEL=/home/aied_test/models/Qwen2.5-VL-7B-Instruct
DATA_DIR=~/ParamMute/data/CoConflictQA
SCRIPT=~/ParamMute/src/3_evaluate/eval_CoConflictQA_qwen2vl.py
LORA=~/ParamMute/checkpoints/qwen2vl_lora_layer12_lambda0.5
DATASETS=(SearchQA_kc.jsonl NaturalQuestionsShort_kc.jsonl)

for DS in "${DATASETS[@]}"; do
    NAME="${DS%.jsonl}"
    CUDA_VISIBLE_DEVICES=0 python $SCRIPT --model_name $MODEL --data_path $DATA_DIR/$DS \
        --act_inhibit_ratio 1.0 --act_inhibit_layer_list 12 --schema instr \
        --output_path ~/ParamMute/results/eval/qwen2vl_baseline/$DS \
        --log_path ~/ParamMute/results/eval/qwen2vl_baseline/${NAME}.log
    CUDA_VISIBLE_DEVICES=0 python $SCRIPT --model_name $MODEL --data_path $DATA_DIR/$DS \
        --act_inhibit_ratio 0.5 --act_inhibit_layer_list 12 --schema instr \
        --output_path ~/ParamMute/results/eval/qwen2vl_layer12_lambda0.5/$DS \
        --log_path ~/ParamMute/results/eval/qwen2vl_layer12_lambda0.5/${NAME}.log
    CUDA_VISIBLE_DEVICES=0 python $SCRIPT --model_name $MODEL --data_path $DATA_DIR/$DS \
        --act_inhibit_ratio 0.5 --act_inhibit_layer_list 12 --lora_path $LORA --schema instr \
        --output_path ~/ParamMute/results/eval/qwen2vl_lora_layer12_lambda0.5/$DS \
        --log_path ~/ParamMute/results/eval/qwen2vl_lora_layer12_lambda0.5/${NAME}.log
done
echo "GPU0 done"
