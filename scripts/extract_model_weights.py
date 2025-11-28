#!/usr/bin/env python3
"""
Extract attention layer weights from a transformer model for use in the MHSA Visualizer.
This script extracts:
- Token embeddings
- Q, K, V projection weights from the first attention layer
- Output projection weights

Usage:
    pip install transformers torch
    python extract_model_weights.py

Output:
    model-weights.json - Contains all extracted weights
"""

import json
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np


def extract_distilbert_weights(model_name="distilbert-base-uncased", layer_idx=0):
    """Extract weights from DistilBERT model."""
    print(f"Loading model: {model_name}")
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    print(f"Model config: embed_dim={config.dim}, num_heads={config.n_heads}")
    
    # Get embedding weights
    embeddings = model.embeddings.word_embeddings.weight.detach().numpy()
    
    # Get attention layer weights
    # DistilBERT uses a single linear layer for Q, K, V combined
    attention = model.transformer.layer[layer_idx].attention
    
    # In DistilBERT, q_lin, k_lin, v_lin are separate linear layers
    Wq = attention.q_lin.weight.detach().numpy()  # [dim, dim]
    Wk = attention.k_lin.weight.detach().numpy()
    Wv = attention.v_lin.weight.detach().numpy()
    Wo = attention.out_lin.weight.detach().numpy()  # Output projection
    
    # Biases
    bq = attention.q_lin.bias.detach().numpy()
    bk = attention.k_lin.bias.detach().numpy()
    bv = attention.v_lin.bias.detach().numpy()
    bo = attention.out_lin.bias.detach().numpy()
    
    print(f"Wq shape: {Wq.shape}")  # Should be [768, 768] for base
    print(f"Wo shape: {Wo.shape}")
    
    return {
        "model_name": model_name,
        "model_type": "distilbert",
        "config": {
            "embed_dim": config.dim,
            "num_heads": config.n_heads,
            "head_dim": config.dim // config.n_heads,
            "vocab_size": config.vocab_size,
            "layer_idx": layer_idx
        },
        "embeddings": embeddings,
        "attention": {
            "Wq": Wq,
            "Wk": Wk,
            "Wv": Wv,
            "Wo": Wo,
            "bq": bq,
            "bk": bk,
            "bv": bv,
            "bo": bo
        },
        "tokenizer_config": {
            "cls_token": tokenizer.cls_token,
            "sep_token": tokenizer.sep_token,
            "pad_token": tokenizer.pad_token,
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }
    }


def extract_bert_weights(model_name="bert-base-uncased", layer_idx=0):
    """Extract weights from BERT model."""
    print(f"Loading model: {model_name}")
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    print(f"Model config: embed_dim={config.hidden_size}, num_heads={config.num_attention_heads}")
    
    # Get embedding weights
    embeddings = model.embeddings.word_embeddings.weight.detach().numpy()
    
    # Get attention layer weights from encoder
    attention = model.encoder.layer[layer_idx].attention.self
    output = model.encoder.layer[layer_idx].attention.output
    
    Wq = attention.query.weight.detach().numpy()
    Wk = attention.key.weight.detach().numpy()
    Wv = attention.value.weight.detach().numpy()
    Wo = output.dense.weight.detach().numpy()
    
    bq = attention.query.bias.detach().numpy()
    bk = attention.key.bias.detach().numpy()
    bv = attention.value.bias.detach().numpy()
    bo = output.dense.bias.detach().numpy()
    
    print(f"Wq shape: {Wq.shape}")
    print(f"Wo shape: {Wo.shape}")
    
    return {
        "model_name": model_name,
        "model_type": "bert",
        "config": {
            "embed_dim": config.hidden_size,
            "num_heads": config.num_attention_heads,
            "head_dim": config.hidden_size // config.num_attention_heads,
            "vocab_size": config.vocab_size,
            "layer_idx": layer_idx
        },
        "embeddings": embeddings,
        "attention": {
            "Wq": Wq,
            "Wk": Wk,
            "Wv": Wv,
            "Wo": Wo,
            "bq": bq,
            "bk": bk,
            "bv": bv,
            "bo": bo
        },
        "tokenizer_config": {
            "cls_token": tokenizer.cls_token,
            "sep_token": tokenizer.sep_token,
            "pad_token": tokenizer.pad_token,
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token_id": tokenizer.sep_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }
    }


def compress_for_web(data, max_vocab_size=30522):
    """Compress the data for web delivery by reducing precision and size."""
    
    def to_list(arr, precision=5):
        """Convert numpy array to list with reduced precision."""
        if isinstance(arr, np.ndarray):
            # Round to reduce JSON size
            return np.round(arr, precision).tolist()
        return arr
    
    compressed = {
        "model_name": data["model_name"],
        "model_type": data["model_type"],
        "config": data["config"],
        "tokenizer_config": data["tokenizer_config"],
        # Full embedding matrix is large (~23MB for BERT), we'll keep it but compress
        "embeddings": to_list(data["embeddings"], precision=4),
        "attention": {
            "Wq": to_list(data["attention"]["Wq"], precision=5),
            "Wk": to_list(data["attention"]["Wk"], precision=5),
            "Wv": to_list(data["attention"]["Wv"], precision=5),
            "Wo": to_list(data["attention"]["Wo"], precision=5),
            "bq": to_list(data["attention"]["bq"], precision=5),
            "bk": to_list(data["attention"]["bk"], precision=5),
            "bv": to_list(data["attention"]["bv"], precision=5),
            "bo": to_list(data["attention"]["bo"], precision=5),
        }
    }
    
    return compressed


def create_mini_model(data, common_tokens=None):
    """Create a mini version with only common token embeddings.
    This dramatically reduces file size while keeping the model functional.
    """
    if common_tokens is None:
        # Common English words + special tokens + common subwords
        common_tokens = list(range(1000)) + list(range(1996, 2100))  # First 1000 + common words
    
    embeddings = np.array(data["embeddings"])
    
    mini_data = {
        "model_name": data["model_name"],
        "model_type": data["model_type"],
        "config": data["config"],
        "tokenizer_config": data["tokenizer_config"],
        "token_ids": common_tokens,  # Which token IDs are included
        "embeddings": embeddings[common_tokens].round(4).tolist(),  # Only included tokens
        "attention": data["attention"]  # Keep full attention weights
    }
    
    return mini_data


def save_json(data, filename, indent=None):
    """Save data to JSON file."""
    print(f"Saving to {filename}...")
    with open(filename, 'w') as f:
        json.dump(data, f, indent=indent)
    
    import os
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Saved {filename}: {size_mb:.2f} MB")


def main():
    import os
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract DistilBERT (smaller, faster)
    print("\n" + "="*60)
    print("Extracting DistilBERT weights...")
    print("="*60)
    
    distilbert_data = extract_distilbert_weights()
    
    # Save full compressed version
    compressed = compress_for_web(distilbert_data)
    save_json(compressed, os.path.join(output_dir, "distilbert-weights-full.json"))
    
    # Save mini version (much smaller, for quick loading)
    mini = create_mini_model(compressed)
    save_json(mini, os.path.join(output_dir, "distilbert-weights.json"))
    
    # Also create an even smaller "attention-only" version
    attention_only = {
        "model_name": distilbert_data["model_name"],
        "model_type": distilbert_data["model_type"],
        "config": distilbert_data["config"],
        "attention": compressed["attention"]
    }
    save_json(attention_only, os.path.join(output_dir, "distilbert-attention.json"))
    
    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - distilbert-weights-full.json (full embeddings, large)")
    print("  - distilbert-weights.json (mini version with common tokens)")
    print("  - distilbert-attention.json (attention weights only, smallest)")


if __name__ == "__main__":
    main()

