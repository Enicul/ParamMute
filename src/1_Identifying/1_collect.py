"""
Phase 1: Activation Collection (server-side only, no visualization).
Collects per-layer activation ratios for each sample and saves to JSONL.
Visualization is done separately on local machine.

Supports: llama3, llama31, qwen25, qwen25vl, gemma3, mistral
"""

import torch
import jsonlines
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file_path", type=str, required=True,
                        help="Input data path (JSONL)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for collected activations (JSONL)")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=['llama3', 'llama31', 'qwen25', 'qwen25vl', 'gemma3', 'mistral'],
                        help="Model type for chat template token detection")
    return parser.parse_args()


class ActivationCollector:
    def __init__(self, model, tokenizer, model_type, num_layers, num_neurons, device, model_backbone=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.device = device
        self.backbone = model_backbone if model_backbone else model.model
        self.reset()

    def reset(self):
        num_layers = self.num_layers
        self.layer_flag = 0
        self.save_flag = False
        self.data_idx = 0

        self.activation_matrix_total = [None] * num_layers
        self.activation_matrix_common = [None] * num_layers
        self.activation_matrix = [None] * num_layers
        self.max_activate = [None] * num_layers
        self.min_activate = [None] * num_layers
        self.avg_activate = [None] * num_layers

    def mlp_forward_hook(self, layer_idx):
        layer_mlp = self.backbone.layers[layer_idx].mlp
        original_mlp_forward = layer_mlp.forward

        def hooked_forward(x):
            if self.save_flag:
                activations = layer_mlp.act_fn(
                    layer_mlp.gate_proj(x)
                )
                actmean = (activations[:, self.data_idx:, :]).sum(dim=1, keepdim=True)
                act_bin = (actmean > 0).squeeze().squeeze()

                if self.activation_matrix_total[layer_idx] is None:
                    self.activation_matrix_total[layer_idx] = act_bin.clone()
                    self.activation_matrix_common[layer_idx] = act_bin.clone()
                    self.activation_matrix[layer_idx] = act_bin.int().clone()
                    self.max_activate[layer_idx] = act_bin.sum()
                    self.min_activate[layer_idx] = act_bin.sum()
                    self.avg_activate[layer_idx] = act_bin.sum()
                else:
                    self.activation_matrix_total[layer_idx] |= act_bin
                    self.activation_matrix_common[layer_idx] &= act_bin
                    self.activation_matrix[layer_idx] += act_bin.int()
                    self.max_activate[layer_idx] = max(act_bin.sum(), self.max_activate[layer_idx])
                    self.min_activate[layer_idx] = min(act_bin.sum(), self.min_activate[layer_idx])
                    self.avg_activate[layer_idx] += act_bin.sum()
                down_proj = layer_mlp.down_proj(
                    activations * layer_mlp.up_proj(x)
                )
                self.layer_flag = (self.layer_flag + 1) % self.num_layers
            else:
                down_proj = original_mlp_forward(x)
            return down_proj
        return hooked_forward

    def attach_hooks(self):
        for layer_idx in range(self.num_layers):
            self.backbone.layers[layer_idx].mlp.forward = self.mlp_forward_hook(layer_idx)

    def find_output_start_idx(self, token_ids):
        sublists = {
            'llama3':   [128006, 78191, 128007, 271],
            'llama31':  [128006, 78191, 128007, 271],
            'llama32':  [128006, 78191, 128007, 271],
            'qwen25':   [151644, 77091, 198],
            'qwen25vl': [151644, 77091, 198],
            'gemma3':   [105, 4368, 107],
            'mistral':  [3],
        }
        if self.model_type == 'mistral':
            sublist = [4]
        else:
            sublist = sublists.get(self.model_type, [])

        start_indices = [
            i for i in range(len(token_ids) - len(sublist) + 1)
            if token_ids[i:i+len(sublist)] == sublist
        ]
        return start_indices[-1] + len(sublist) if start_indices else 0

    def evaluate_per_example(self, in_file_path, output_path):
        self.attach_hooks()
        cnt = 0

        with jsonlines.open(in_file_path) as reader:
            all_datas = list(reader)

        with jsonlines.open(output_path, mode='w') as writer:
            for data in tqdm(all_datas, desc=f'Collecting activations ({self.model_type})'):
                cnt += 1
                cur_output = data['pred']
                cur_input_w_context = data['prompt_w_context']

                modes = {
                    'ww': [
                        {'role': 'user', 'content': cur_input_w_context},
                        {'role': 'assistant', 'content': cur_output}
                    ],
                }
                per_data_results = {}

                for mode_name, cur_data in modes.items():
                    self.reset()

                    cur_data_tokens = self.tokenizer.apply_chat_template(cur_data, tokenize=False)
                    cur_data_ids = self.tokenizer.apply_chat_template(cur_data, tokenize=True)
                    self.data_idx = self.find_output_start_idx(cur_data_ids)
                    input_ids = self.tokenizer(cur_data_tokens, return_tensors='pt').to(self.device)
                    self.save_flag = True
                    with torch.inference_mode():
                        self.model(**input_ids)

                    if cnt % 2000 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                    avg_activations_per_layer = [
                        (m.item() / self.num_neurons) if m is not None else None
                        for m in self.avg_activate
                    ]
                    per_data_results[mode_name] = avg_activations_per_layer

                out_record = {
                    'question': data.get('question', ''),
                    'is_faithful': data.get('is_faithful', -1),
                    'activations_avg': per_data_results,
                }
                writer.write(out_record)

        print(f"Done. Saved {cnt} records to {output_path}")


def load_model(model_path, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )

    # Gemma-3 nests the text backbone under model.language_model
    if model_type == 'gemma3':
        backbone = model.model.language_model
        num_layers = model.config.text_config.num_hidden_layers
        num_neurons = model.config.text_config.intermediate_size
    else:
        backbone = model.model
        num_layers = model.config.num_hidden_layers
        num_neurons = model.config.intermediate_size

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loaded {model_type}: {num_layers} layers, {num_neurons} neurons/layer")
    return model, tokenizer, num_layers, num_neurons, device, backbone


if __name__ == "__main__":
    args = get_args()

    model, tokenizer, num_layers, num_neurons, device, backbone = load_model(
        args.pretrained_model_path, args.model_type
    )

    collector = ActivationCollector(
        model, tokenizer, args.model_type,
        num_layers, num_neurons, device,
        model_backbone=backbone
    )

    collector.evaluate_per_example(args.in_file_path, args.output_path)
