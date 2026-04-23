"""
SFT + FFN suppression for Qwen2.5-VL-7B-Instruct.
Loads with system transformers, patches layer 12 MLP forward, adds LoRA, trains on pip_kag data.
"""

import os
import types
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── Suppression ───────────────────────────────────────────────────────────────

def apply_ffn_suppression(model, inhibit_strength: float, inhibit_layer_list: list):
    if inhibit_strength >= 1.0:
        logger.info("inhibit_strength=1.0, no suppression applied (baseline).")
        return model
    patched = []
    for name, module in model.named_modules():
        parts = name.split('.')
        if parts[-1] == 'mlp' and len(parts) >= 2 and parts[-2].isdigit() and 'visual' not in name:
            layer_idx = int(parts[-2])
            if layer_idx in inhibit_layer_list:
                orig_forward = module.forward

                def make_patched(orig, strength):
                    def patched(x):
                        return orig(x) * strength
                    return patched

                module.forward = types.MethodType(
                    lambda self, x, _f=make_patched(orig_forward, inhibit_strength): _f(x),
                    module
                )
                patched.append(f"{name} (layer {layer_idx})")
    if not patched:
        raise RuntimeError(f"No MLP modules found for layers {inhibit_layer_list}")
    logger.info(f"Suppressed (strength={inhibit_strength}): {patched}")
    return model


# ── Dataset ───────────────────────────────────────────────────────────────────

class QwenSFTDataset(Dataset):
    USER_FORMAT = (
        '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
        '<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n'
    )
    ASSISTANT_FORMAT = '{content}<|im_end|>\n'

    def __init__(self, data_path: str, tokenizer, max_len: int):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        human = self.USER_FORMAT.format(content=data['rag_input'])
        assistant = self.ASSISTANT_FORMAT.format(content=data['output'])

        input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

        input_ids = (input_tokens + output_tokens)[:self.max_len]
        labels = ([-100] * len(input_tokens) + output_tokens)[:self.max_len]
        attention_mask = [1] * len(input_ids)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


class SFTDataCollator:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lengths = [len(x['input_ids']) for x in batch]
        batch_max = min(max(lengths), self.max_seq_length)

        input_ids_batch, attn_batch, labels_batch = [], [], []
        for x in batch:
            pad_len = batch_max - len(x['input_ids'])
            input_ids_batch.append((x['input_ids'] + [self.pad_token_id] * pad_len)[:self.max_seq_length])
            attn_batch.append((x['attention_mask'] + [0] * pad_len)[:self.max_seq_length])
            labels_batch.append((x['labels'] + [-100] * pad_len)[:self.max_seq_length])

        return {
            'input_ids': torch.tensor(input_ids_batch, dtype=torch.long),
            'attention_mask': torch.tensor(attn_batch, dtype=torch.long),
            'labels': torch.tensor(labels_batch, dtype=torch.long),
        }


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelArgs:
    model_name_or_path: str = field(metadata={"help": "Path to Qwen2.5-VL model"})
    inhibit_strength: float = field(default=0.5)
    inhibit_layer_list: List[int] = field(default_factory=lambda: [12])
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)

@dataclass
class DataArgs:
    train_file: str = field(metadata={"help": "Path to pip_kag_train.jsonl"})
    max_len: int = field(default=1024)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = HfArgumentParser((ModelArgs, DataArgs, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model — load with system transformers
    logger.info(f"Loading Qwen2.5-VL from {model_args.model_name_or_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        device_map=None,  # Trainer handles device placement with LoRA
    )

    # Suppress FFN at target layers
    model = apply_ffn_suppression(model, model_args.inhibit_strength, model_args.inhibit_layer_list)

    # Freeze vision tower, apply LoRA to language backbone only
    for name, param in model.named_parameters():
        if 'visual' in name:
            param.requires_grad = False

    lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=lora_target_modules,
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_dataset = QwenSFTDataset(data_args.train_file, tokenizer, data_args.max_len)
    collator = SFTDataCollator(tokenizer, data_args.max_len)
    logger.info(f"Training on {len(train_dataset)} samples")

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == '__main__':
    main()
