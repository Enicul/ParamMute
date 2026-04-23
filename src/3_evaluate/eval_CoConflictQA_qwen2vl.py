import re
import ast
import string
import json
import jsonlines
import argparse
from rouge import Rouge
from tqdm import tqdm
import os
import torch
from collections import Counter
import logging

import types
# Use system transformers (supports qwen2_5_vl) — local ParamMute transformers is too old
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


def apply_ffn_suppression(model, inhibit_strength: float, inhibit_layer_list: list):
    """Patch language model MLP forward at target layers. Path: model.model.language_model.layers"""
    if inhibit_strength == 1.0:
        return model  # no-op for baseline
    lm_layers = model.model.language_model.layers
    for layer_idx in inhibit_layer_list:
        mlp = lm_layers[layer_idx].mlp
        orig_forward = mlp.forward

        def make_patched(orig, strength):
            def patched(x):
                return orig(x) * strength
            return patched

        mlp.forward = types.MethodType(
            lambda self, x, _f=make_patched(orig_forward, inhibit_strength): _f(x),
            mlp
        )
        print(f"[ParamMute] Layer {layer_idx} LM MLP suppressed (strength={inhibit_strength})")
    return model

print('=' * 20 + f' GPUs: {torch.cuda.device_count()} ' + '=' * 20)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def call_model(model, tokenizer, input_ids, max_new_tokens):
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )[0, input_ids.shape[-1]:]
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _acc_score(prediction, ground_truth):
    return 1.0 if normalize_answer(ground_truth) in normalize_answer(prediction) else 0.0


def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(normalize_answer(prediction), normalize_answer(ground_truth), avg=True)
    except ValueError:
        return 0.0
    return scores["rouge-l"]["f"]


def _f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def get_score_with_parametric(parametric_answers, preds, golds):
    pm_acc, acc, rouge, f1, em = 0, 0, 0, 0, 0
    for pm_answer, pred, gold in zip(parametric_answers, preds, golds):
        glist = gold if isinstance(gold, list) else [gold]
        _acc = max(_acc_score(pred, g) for g in glist)
        _rouge = max(_rougel_score(pred, g) for g in glist)
        _f1 = max(_f1_score(pred, g) for g in glist)
        _em = max(_exact_match_score(pred, g) for g in glist)
        _pm = _acc_score(pred, pm_answer)
        if _pm == 1:
            _acc = 0
        pm_acc += _pm
        acc += _acc
        rouge += _rouge
        f1 += _f1
        em += _em
    n = len(preds) + 1e-5
    return (pm_acc * 100 / n, acc * 100 / n, rouge * 100 / n, f1 * 100 / n, em * 100 / n)


def eval_step(parametric_answers, pred_answers, gold_answers, step):
    pm_acc, acc, rouge, f1, em = get_score_with_parametric(parametric_answers, pred_answers, gold_answers)
    mr = round((pm_acc / (pm_acc + acc + 1e-5)) * 100, 2)
    logging.info('Step {}: context_acc={:.2f} pm_acc={:.2f} mr={:.2f} em={:.2f}'.format(
        step, acc, pm_acc, mr, em))
    return acc, pm_acc, mr, em


def build_prompt(query, context, schema, tokenizer, use_chat_template):
    if schema == 'base_wo_context':
        text = f'Q: {query}\nA: '
    elif schema == 'base':
        text = f'{context}\nQ: {query}\nA: '
    elif schema == 'opin':
        context = context.replace('"', '')
        text = f'Bob said "{context}"\nQ: {query[:-1]} in Bob\'s opinion?\nA: '
    elif schema == 'instr+opin':
        context = context.replace('"', '')
        text = f'Bob said "{context}"\nQ: {query[:-1]} in Bob\'s opinion?\nA:'
    elif schema == 'attr':
        text = f'{context}\nQ: {query[:-1]} based on the given text?\nA:'
    elif schema == 'instr':
        text = f'{context}\nQ: {query}\nA: '
    else:
        text = f'{context}\nQ: {query}\nA: '

    if use_chat_template:
        messages = [{'role': 'user', 'content': text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='./Models/Qwen2.5-VL-7B-Instruct', type=str)
    parser.add_argument('--data_path', default='./data/CoConflictQA/CoConflictQA.jsonl', type=str)
    parser.add_argument('--schema', default='instr', type=str,
                        help='base | base_wo_context | attr | instr | opin | instr+opin')
    parser.add_argument('--output_path', default='./result/qwen2vl_suppressed.jsonl', type=str)
    parser.add_argument('--log_path', default='./log/qwen2vl_suppressed.log', type=str)
    parser.add_argument('--use_chat_template', type=str2bool, nargs='?', default=True)
    parser.add_argument('--max_new_tokens', default=32, type=int)
    parser.add_argument('--act_inhibit_ratio', default=0.5, type=float,
                        help='FFN suppression strength (0=full suppress, 1=no-op). Default 0.5.')
    parser.add_argument('--act_inhibit_layer_list', type=int, nargs='+', default=[12],
                        help='Decoder layer indices to suppress. Default [12] (best TPC layer).')
    parser.add_argument('--lora_path', default=None, type=str,
                        help='Path to LoRA adapter checkpoint. If None, runs without LoRA.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(args.log_path), logging.StreamHandler()],
    )

    logging.info(f'Model: {args.model_name}')
    logging.info(f'inhibit_strength={args.act_inhibit_ratio}, layers={args.act_inhibit_layer_list}')
    logging.info(f'lora_path={args.lora_path}')
    logging.info(f'schema={args.schema}, use_chat_template={args.use_chat_template}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        device_map='auto',
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
    )
    model = apply_ffn_suppression(model, args.act_inhibit_ratio, args.act_inhibit_layer_list)
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        logging.info(f'LoRA adapter loaded from {args.lora_path}')
    model.eval()

    with jsonlines.open(args.data_path, 'r') as reader:
        data = list(reader)
    logging.info(f'Loaded {len(data)} instances.')

    # Pre-tokenize all prompts
    input_ids_list = []
    for d in data:
        prompt = build_prompt(
            d['question'], d['context'], args.schema, tokenizer, args.use_chat_template
        )
        input_ids_list.append(
            tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
        )

    gold_answers, pred_answers, pm_answers = [], [], []
    for idx, d in tqdm(enumerate(data), total=len(data)):
        gold_answers.append(d['answers'])
        pm_answers.append(d['parametric_answer'])

        pred = call_model(model, tokenizer, input_ids_list[idx], args.max_new_tokens)
        pred_answers.append(pred)
        d['pred'] = pred

        if (idx + 1) % 500 == 0:
            eval_step(pm_answers, pred_answers, gold_answers, idx + 1)

    logging.info('Final evaluation...')
    final_acc, final_pm, final_mr, final_em = eval_step(pm_answers, pred_answers, gold_answers, len(data))
    logging.info(f'context_acc={final_acc:.2f} pm_acc={final_pm:.2f} mr={final_mr:.2f} em={final_em:.2f}')

    with jsonlines.open(args.output_path, mode='w') as writer:
        for d in data:
            writer.write(d)
    logging.info(f'Results saved to {args.output_path}')


if __name__ == '__main__':
    main()
