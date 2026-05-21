import argparse
import jsonlines
import logging
import os
import re
import string
import types
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration
from peft import PeftModel

try:
    from rouge import Rouge
except ImportError:
    Rouge = None


def apply_ffn_suppression(model, inhibit_strength: float, inhibit_layer_list: list):
    if inhibit_strength == 1.0:
        return model

    patched = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if parts[-1] == "mlp" and len(parts) >= 2 and parts[-2].isdigit():
            if "vision_model" in name or "vision" in name:
                continue
            layer_idx = int(parts[-2])
            if layer_idx in inhibit_layer_list:
                orig_forward = module.forward

                def make_patched(orig, strength):
                    def patched_forward(*args, **kwargs):
                        return orig(*args, **kwargs) * strength
                    return patched_forward

                module.forward = types.MethodType(
                    lambda self, *args, _f=make_patched(orig_forward, inhibit_strength), **kwargs: _f(*args, **kwargs),
                    module,
                )
                patched.append(f"{name} (layer {layer_idx})")

    if not patched:
        raise RuntimeError(f"No language MLP modules found for layers {inhibit_layer_list}")
    logging.info("[ParamMute] Suppressed strength=%s modules=%s", inhibit_strength, patched)
    return model


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def _acc_score(prediction, ground_truth):
    return 1.0 if normalize_answer(ground_truth) in normalize_answer(prediction) else 0.0


def _rougel_score(prediction, ground_truth):
    if Rouge is None:
        return 0.0
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
        return 0.0
    precision = num_same / max(len(pred_tokens), 1)
    recall = num_same / max(len(gold_tokens), 1)
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
    logging.info(
        "Step %s: context_acc=%.2f pm_acc=%.2f mr=%.2f em=%.2f rouge=%.2f f1=%.2f",
        step, acc, pm_acc, mr, em, rouge, f1,
    )
    return acc, pm_acc, mr, em


def build_prompt(query, context, schema):
    if schema == "base_wo_context":
        text = f"Q: {query}\nA: "
    elif schema == "base":
        text = f"{context}\nQ: {query}\nA: "
    elif schema == "opin":
        context = context.replace('"', "")
        text = f"Bob said \"{context}\"\nQ: {query[:-1]} in Bob's opinion?\nA: "
    elif schema == "instr+opin":
        context = context.replace('"', "")
        text = f"Bob said \"{context}\"\nQ: {query[:-1]} in Bob's opinion?\nA:"
    elif schema == "attr":
        text = f"{context}\nQ: {query[:-1]} based on the given text?\nA:"
    else:
        text = f"{context}\nQ: {query}\nA: "

    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return messages


def call_model(model, processor, messages, max_new_tokens):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )[0, inputs["input_ids"].shape[-1]:]
    return processor.decode(out, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="/home/aied_test/models/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--schema", default="instr")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--act_inhibit_ratio", default=1.0, type=float)
    parser.add_argument("--act_inhibit_layer_list", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--lora_path", default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(args.log_path), logging.StreamHandler()],
    )

    logging.info("Model: %s", args.model_name)
    logging.info("inhibit_strength=%s layers=%s", args.act_inhibit_ratio, args.act_inhibit_layer_list)
    logging.info("lora_path=%s", args.lora_path)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model = apply_ffn_suppression(model, args.act_inhibit_ratio, args.act_inhibit_layer_list)
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        logging.info("LoRA adapter loaded from %s", args.lora_path)
    model.eval()

    with jsonlines.open(args.data_path, "r") as reader:
        data = list(reader)
    logging.info("Loaded %d instances.", len(data))

    gold_answers, pred_answers, pm_answers = [], [], []
    for idx, d in tqdm(enumerate(data), total=len(data)):
        gold_answers.append(d["answers"])
        pm_answers.append(d["parametric_answer"])
        pred = call_model(model, processor, build_prompt(d["question"], d["context"], args.schema), args.max_new_tokens)
        pred_answers.append(pred)
        d["pred"] = pred
        if (idx + 1) % 500 == 0:
            eval_step(pm_answers, pred_answers, gold_answers, idx + 1)

    final_acc, final_pm, final_mr, final_em = eval_step(pm_answers, pred_answers, gold_answers, len(data))
    logging.info("FINAL context_acc=%.2f pm_acc=%.2f mr=%.2f em=%.2f", final_acc, final_pm, final_mr, final_em)

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(data)
    logging.info("Results saved to %s", args.output_path)


if __name__ == "__main__":
    main()
