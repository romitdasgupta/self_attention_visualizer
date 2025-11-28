#!/usr/bin/env python3
"""
Validate MHSA Visualizer attention outputs against HuggingFace DistilBERT.

This script:
1. Runs DistilBERT on an input sequence with output_attentions=True
2. Compares the attention patterns from layer 0 heads
3. Reports known patterns for layer 0 heads

Usage:
    pip install transformers torch numpy
    python validate_attention.py "The cat sat on the mat"
"""

import sys
import json
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel


def analyze_attention_patterns(attention_weights, tokens):
    """
    Analyze attention patterns and identify common behaviors.
    
    Args:
        attention_weights: numpy array of shape [num_heads, seq_len, seq_len]
        tokens: list of token strings
    
    Returns:
        dict with analysis for each head
    """
    num_heads, seq_len, _ = attention_weights.shape
    analysis = {}
    
    for h in range(num_heads):
        head_attn = attention_weights[h]
        head_analysis = {
            "head_idx": h,
            "patterns": []
        }
        
        # Check for [CLS] attention (first token gets high attention)
        cls_attention = head_attn[:, 0].mean()
        if cls_attention > 0.3:
            head_analysis["patterns"].append(f"Attends to [CLS] (avg: {cls_attention:.2f})")
        
        # Check for [SEP] attention (last token gets high attention)
        sep_attention = head_attn[:, -1].mean()
        if sep_attention > 0.3:
            head_analysis["patterns"].append(f"Attends to [SEP] (avg: {sep_attention:.2f})")
        
        # Check for self-attention (diagonal is high)
        self_attention = np.diag(head_attn).mean()
        if self_attention > 0.3:
            head_analysis["patterns"].append(f"Self-attention (avg: {self_attention:.2f})")
        
        # Check for previous token attention (attending to token i-1)
        prev_attention = np.mean([head_attn[i, i-1] for i in range(1, seq_len)])
        if prev_attention > 0.2:
            head_analysis["patterns"].append(f"Previous token (avg: {prev_attention:.2f})")
        
        # Check for next token attention (attending to token i+1)
        next_attention = np.mean([head_attn[i, i+1] for i in range(seq_len-1)])
        if next_attention > 0.2:
            head_analysis["patterns"].append(f"Next token (avg: {next_attention:.2f})")
        
        # Check for broad/uniform attention
        entropy = -np.sum(head_attn * np.log(head_attn + 1e-10), axis=1).mean()
        max_entropy = np.log(seq_len)
        if entropy > 0.8 * max_entropy:
            head_analysis["patterns"].append(f"Broad attention (entropy: {entropy:.2f}/{max_entropy:.2f})")
        
        # Compute summary statistics
        head_analysis["stats"] = {
            "cls_attention": float(cls_attention),
            "sep_attention": float(sep_attention),
            "self_attention": float(self_attention),
            "prev_token_attention": float(prev_attention),
            "next_token_attention": float(next_attention),
            "entropy_ratio": float(entropy / max_entropy)
        }
        
        analysis[f"head_{h}"] = head_analysis
    
    return analysis


def run_distilbert(input_text):
    """
    Run DistilBERT and extract layer 0 attention weights.
    """
    print(f"Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=True)
    model.eval()
    
    print(f"\nInput text: '{input_text}'")
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get layer 0 attention (shape: [batch, num_heads, seq_len, seq_len])
    layer_0_attn = outputs.attentions[0][0].numpy()  # Remove batch dimension
    
    print(f"\nLayer 0 attention shape: {layer_0_attn.shape}")
    print(f"Number of heads: {layer_0_attn.shape[0]}")
    
    return {
        "tokens": tokens,
        "attention_weights": layer_0_attn,
        "input_ids": inputs['input_ids'][0].tolist()
    }


def print_attention_matrix(attn_matrix, tokens, head_idx):
    """Print a formatted attention matrix."""
    print(f"\n{'='*60}")
    print(f"Head {head_idx} Attention Matrix")
    print(f"{'='*60}")
    
    # Header
    header = "         " + " ".join([f"{t[:6]:>7}" for t in tokens])
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, token in enumerate(tokens):
        row = f"{token[:8]:>8} " + " ".join([f"{attn_matrix[i,j]:7.3f}" for j in range(len(tokens))])
        print(row)


def main():
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "The cat sat on the mat"
    
    # Run DistilBERT
    result = run_distilbert(input_text)
    tokens = result["tokens"]
    attention_weights = result["attention_weights"]
    
    # Analyze patterns
    print("\n" + "="*60)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*60)
    
    analysis = analyze_attention_patterns(attention_weights, tokens)
    
    for head_name, head_analysis in analysis.items():
        print(f"\n{head_name}:")
        if head_analysis["patterns"]:
            for pattern in head_analysis["patterns"]:
                print(f"  ✓ {pattern}")
        else:
            print("  No strong patterns detected")
        
        stats = head_analysis["stats"]
        print(f"  Stats: CLS={stats['cls_attention']:.2f}, SEP={stats['sep_attention']:.2f}, "
              f"Self={stats['self_attention']:.2f}, Prev={stats['prev_token_attention']:.2f}, "
              f"Next={stats['next_token_attention']:.2f}")
    
    # Print first few attention matrices
    print("\n" + "="*60)
    print("ATTENTION MATRICES (first 4 heads)")
    print("="*60)
    
    for h in range(min(4, attention_weights.shape[0])):
        print_attention_matrix(attention_weights[h], tokens, h)
    
    # Save results for comparison with JS visualizer
    output = {
        "input_text": input_text,
        "tokens": tokens,
        "attention_weights": attention_weights.tolist(),
        "analysis": analysis
    }
    
    with open("validation_output.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to validation_output.json")
    print("\nTo validate against the MHSA Visualizer:")
    print("1. Open the visualizer in a browser")
    print("2. Enter the same input text")
    print("3. Select the DistilBERT model with matching number of heads")
    print("4. Compare the attention patterns shown in the 'Individual Heads' tab")


if __name__ == "__main__":
    main()

