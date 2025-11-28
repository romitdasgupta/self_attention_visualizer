#!/usr/bin/env python3
"""
Create a compact version of model weights for web delivery.
Reduces precision and optionally uses only a subset of heads.
"""

import json
import numpy as np
import os
import gzip


def load_attention_weights(filepath):
    """Load the attention-only weights file."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        return json.load(f)


def reshape_to_heads(W, num_heads, head_dim):
    """Reshape a [embed_dim, embed_dim] weight matrix to per-head weights [num_heads, embed_dim, head_dim]."""
    W = np.array(W)
    embed_dim = W.shape[0]
    # W is [embed_dim, embed_dim] = [embed_dim, num_heads * head_dim]
    # Reshape to [embed_dim, num_heads, head_dim]
    W_reshaped = W.reshape(embed_dim, num_heads, head_dim)
    # Transpose to [num_heads, embed_dim, head_dim]
    return W_reshaped.transpose(1, 0, 2)


def reshape_bias_to_heads(b, num_heads, head_dim):
    """Reshape bias [embed_dim] to per-head [num_heads, head_dim]."""
    b = np.array(b)
    return b.reshape(num_heads, head_dim)


def create_compact_version(data, num_heads_to_keep=4, precision=3):
    """Create a compact version with fewer heads and lower precision."""
    config = data["config"]
    orig_num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    embed_dim = config["embed_dim"]
    
    print(f"Original: {orig_num_heads} heads, {head_dim} head_dim, {embed_dim} embed_dim")
    print(f"Keeping first {num_heads_to_keep} heads with {precision} decimal precision")
    
    # Reshape weights to per-head format
    Wq_heads = reshape_to_heads(data["attention"]["Wq"], orig_num_heads, head_dim)
    Wk_heads = reshape_to_heads(data["attention"]["Wk"], orig_num_heads, head_dim)
    Wv_heads = reshape_to_heads(data["attention"]["Wv"], orig_num_heads, head_dim)
    
    bq_heads = reshape_bias_to_heads(data["attention"]["bq"], orig_num_heads, head_dim)
    bk_heads = reshape_bias_to_heads(data["attention"]["bk"], orig_num_heads, head_dim)
    bv_heads = reshape_bias_to_heads(data["attention"]["bv"], orig_num_heads, head_dim)
    
    # For Wo, we need to handle differently
    # Original Wo is [embed_dim, embed_dim] = [num_heads * head_dim, embed_dim]
    # We need [num_heads, head_dim, embed_dim]
    Wo = np.array(data["attention"]["Wo"])
    Wo_heads = Wo.reshape(orig_num_heads, head_dim, embed_dim)
    
    # Keep only first N heads
    heads = []
    for h in range(num_heads_to_keep):
        heads.append({
            "Wq": np.round(Wq_heads[h], precision).tolist(),
            "Wk": np.round(Wk_heads[h], precision).tolist(),
            "Wv": np.round(Wv_heads[h], precision).tolist(),
            "bq": np.round(bq_heads[h], precision).tolist(),
            "bk": np.round(bk_heads[h], precision).tolist(),
            "bv": np.round(bv_heads[h], precision).tolist(),
            "Wo": np.round(Wo_heads[h], precision).tolist()
        })
    
    # Output bias is shared
    bo = np.round(np.array(data["attention"]["bo"]), precision).tolist()
    
    compact = {
        "model_name": data["model_name"],
        "model_type": data["model_type"],
        "config": {
            "embed_dim": embed_dim,
            "num_heads": num_heads_to_keep,
            "head_dim": head_dim,
            "original_num_heads": orig_num_heads
        },
        "heads": heads,
        "bo": bo  # Shared output bias
    }
    
    return compact


def create_tiny_demo_version(embed_dim=64, num_heads=4, head_dim=16):
    """Create a tiny demonstration version with random but coherent weights.
    This version is small enough for instant loading but won't show real model behavior.
    Good for development/testing.
    """
    np.random.seed(42)  # Reproducible
    
    heads = []
    for h in range(num_heads):
        # Xavier initialization for more realistic weights
        scale_qkv = np.sqrt(2.0 / (embed_dim + head_dim))
        scale_o = np.sqrt(2.0 / (head_dim + embed_dim))
        
        heads.append({
            "Wq": (np.random.randn(embed_dim, head_dim) * scale_qkv).round(4).tolist(),
            "Wk": (np.random.randn(embed_dim, head_dim) * scale_qkv).round(4).tolist(),
            "Wv": (np.random.randn(embed_dim, head_dim) * scale_qkv).round(4).tolist(),
            "bq": (np.zeros(head_dim)).tolist(),
            "bk": (np.zeros(head_dim)).tolist(),
            "bv": (np.zeros(head_dim)).tolist(),
            "Wo": (np.random.randn(head_dim, embed_dim) * scale_o).round(4).tolist()
        })
    
    return {
        "model_name": "demo-tiny",
        "model_type": "demo",
        "config": {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "head_dim": head_dim
        },
        "heads": heads,
        "bo": [0.0] * embed_dim
    }


def add_embedding_lookup(compact_data, embeddings_data, common_words=None):
    """Add a small embedding lookup table for common words."""
    # Define common words to include in the mini embedding table
    if common_words is None:
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
            "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other", "some", "such",
            "cat", "sat", "mat", "dog", "man", "woman", "child", "house", "car", "tree",
            "love", "like", "want", "think", "know", "see", "look", "use", "find", "give",
            "tell", "say", "make", "go", "take", "come", "get", "put", "read", "write",
            "hello", "world", "good", "bad", "new", "old", "first", "last", "long", "short",
            "big", "small", "high", "low", "young", "old", "great", "little", "own", "other",
            "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"
        ]
    
    # This would require having the tokenizer to get token IDs
    # For now, we'll skip embedding lookup and rely on transformers.js tokenizer
    return compact_data


def save_compact(data, filepath):
    """Save compact data as JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, separators=(',', ':'))  # No whitespace
    
    size_kb = os.path.getsize(filepath) / 1024
    print(f"Saved {filepath}: {size_kb:.1f} KB")
    
    # Also save gzipped version
    gz_path = filepath + '.gz'
    with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    
    gz_size_kb = os.path.getsize(gz_path) / 1024
    print(f"Saved {gz_path}: {gz_size_kb:.1f} KB (gzipped)")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    # Create tiny demo version first (instant loading)
    print("\n" + "="*60)
    print("Creating tiny demo weights...")
    print("="*60)
    tiny = create_tiny_demo_version(embed_dim=64, num_heads=4, head_dim=16)
    save_compact(tiny, os.path.join(output_dir, "demo-tiny-weights.json"))
    
    # Check if full weights exist
    full_weights_path = os.path.join(output_dir, "distilbert-attention.json")
    if os.path.exists(full_weights_path):
        print("\n" + "="*60)
        print("Creating compact DistilBERT weights...")
        print("="*60)
        
        data = load_attention_weights(full_weights_path)
        
        # Create 4-head compact version (most common use case)
        compact_4head = create_compact_version(data, num_heads_to_keep=4, precision=3)
        save_compact(compact_4head, os.path.join(output_dir, "distilbert-4head.json"))
        
        # Create 2-head version (even smaller)
        compact_2head = create_compact_version(data, num_heads_to_keep=2, precision=3)
        save_compact(compact_2head, os.path.join(output_dir, "distilbert-2head.json"))
        
        # Create single-head version (minimal)
        compact_1head = create_compact_version(data, num_heads_to_keep=1, precision=3)
        save_compact(compact_1head, os.path.join(output_dir, "distilbert-1head.json"))
    else:
        print(f"\nWarning: {full_weights_path} not found. Run extract_model_weights.py first.")
    
    print("\n" + "="*60)
    print("Done! Available model weight files:")
    print("="*60)
    for f in os.listdir(output_dir):
        if f.endswith('.json'):
            size = os.path.getsize(os.path.join(output_dir, f)) / 1024
            print(f"  {f}: {size:.1f} KB")


if __name__ == "__main__":
    main()

