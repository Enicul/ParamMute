"""
UA-FFN Layer Selection with Normalization
-----------------------------------------
Implements two normalization approaches from feedback:
  1. Relative gap:  ΔR̃^l = (R^l_unfaithful - R^l_faithful) / E[R^l]
  2. Z-score gap:   ΔR̂^l = (ΔR^l - mean(ΔR)) / std(ΔR)

Then selects Top-K layers by each method per model.
"""

import json
import os
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/func_data/activations"
TOP_K = 6          # number of UA-FFN layers to select
ACT_KEY = "ww"    # key inside activations_avg dict

MODEL_FILES = {
    "LLaMA-3.1-8B":  "activations_llama31_8b.jsonl",
    "LLaMA-3.1-70B": "activations_llama31_70b.jsonl",
    "Qwen2.5-7B":    "activations_qwen25_7b.jsonl",
    "Qwen3-14B":     "activations_qwen3_14b.jsonl",
    "Gemma-3-12B":   "activations_gemma3_12b.jsonl",
    "Mistral-7B":    "activations_mistral_7b.jsonl",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_activations(path):
    faithful, unfaithful = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            acts = d["activations_avg"][ACT_KEY]
            if d["is_faithful"] == 1:
                faithful.append(acts)
            else:
                unfaithful.append(acts)
    return np.array(faithful), np.array(unfaithful)


def compute_delta(faithful, unfaithful):
    """Raw ΔR^l = mean_unfaithful - mean_faithful per layer."""
    R_faithful   = faithful.mean(axis=0)
    R_unfaithful = unfaithful.mean(axis=0)
    delta        = R_unfaithful - R_faithful
    return delta, R_faithful, R_unfaithful


def relative_gap(delta, faithful, unfaithful):
    """ΔR̃^l = ΔR^l / E[R^l]  (mean activation ratio across all samples & layers)."""
    all_acts = np.concatenate([faithful, unfaithful], axis=0)
    mean_R   = all_acts.mean()
    return delta / (mean_R + 1e-10)


def zscore_gap(delta):
    """ΔR̂^l = (ΔR^l - mean(ΔR)) / std(ΔR)  across layers."""
    return (delta - delta.mean()) / (delta.std() + 1e-10)


def top_k_layers(scores, k):
    """Return indices of top-k layers sorted ascending."""
    return sorted(np.argsort(scores)[::-1][:k].tolist())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"{'Model':<20} {'Method':<12} {'Top-K Layers':<35} {'Scores (rounded)'}")
    print("=" * 90)

    results = {}

    for model_name, fname in MODEL_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"{model_name:<20} FILE NOT FOUND: {path}")
            continue

        faithful, unfaithful = load_activations(path)
        delta, R_f, R_u = compute_delta(faithful, unfaithful)

        rel   = relative_gap(delta, faithful, unfaithful)
        zsc   = zscore_gap(delta)

        top_raw = top_k_layers(delta, TOP_K)
        top_rel = top_k_layers(rel,   TOP_K)
        top_zsc = top_k_layers(zsc,   TOP_K)

        n_layers = len(delta)
        results[model_name] = {
            "n_layers":   n_layers,
            "top_k":      TOP_K,
            # Top-K selections
            "raw":        top_raw,
            "relative":   top_rel,
            "zscore":     top_zsc,
            # Full per-layer scores (all layers, for visualization)
            "per_layer": [
                {
                    "layer":          l,
                    "R_faithful":     float(R_f[l]),
                    "R_unfaithful":   float(R_u[l]),
                    "delta":          float(delta[l]),
                    "relative_gap":   float(rel[l]),
                    "zscore_gap":     float(zsc[l]),
                }
                for l in range(n_layers)
            ],
        }

        for method, layers, scores in [
            ("raw ΔR",    top_raw, delta),
            ("relative",  top_rel, rel),
            ("z-score",   top_zsc, zsc),
        ]:
            score_str = " ".join(f"{scores[l]:.3f}" for l in layers)
            print(f"{model_name:<20} {method:<12} {str(layers):<35} {score_str}")

        # Print full layer table
        print(f"  {'Layer':<8} {'R_faithful':<14} {'R_unfaithful':<16} {'ΔR':<12} {'Relative':<12} {'Z-score'}")
        for l in range(n_layers):
            print(f"  {l:<8} {R_f[l]:<14.5f} {R_u[l]:<16.5f} {delta[l]:<12.5f} {rel[l]:<12.5f} {zsc[l]:.5f}")
        print()

    # Save results
    out_path = "./results/ua_ffn_normalized.json"
    os.makedirs("./results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print suppression commands
    print("\n" + "=" * 90)
    print("Suggested suppression layer lists (z-score, Top-K):")
    print("=" * 90)
    for model_name, res in results.items():
        layers_str = " ".join(str(l) for l in res["zscore"])
        print(f"{model_name:<20}: --act_inhibit_layer_list {layers_str}")


if __name__ == "__main__":
    main()
