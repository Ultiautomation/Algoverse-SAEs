"""
analyze_layer18.py
==================

Deep dive into what makes Layer 18 features causally effective for refusal steering
compared to layers 19-20 which only show correlation but no causal effect.

Analysis:
1. Feature activation patterns across layers
2. Top activating examples for each layer's best feature  
3. Feature vector similarity analysis
4. Activation distribution analysis
"""

import torch
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
WIDTH_TAG = "width_16k/canonical"

def analyze_layer_differences():
    """Compare layers 18, 19, 20 to understand why 18 works for steering"""
    
    # Load model
    model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    
    # Load test data
    harmful_ds = load_dataset("walledai/AdvBench", split="train")
    harmful_prompts = harmful_ds["prompt"][:50]  # Smaller sample for analysis
    
    harmless_prompts = [
        "What is the capital of France?",
        "How do you make a sandwich?", 
        "Explain photosynthesis.",
        "What are the benefits of exercise?",
        "How does a computer work?",
    ] * 10
    
    prompts = harmful_prompts + harmless_prompts
    labels = torch.cat([
        torch.ones(len(harmful_prompts), dtype=torch.bool, device=DEVICE),
        torch.zeros(len(harmless_prompts), dtype=torch.bool, device=DEVICE)
    ])
    
    # Load results from previous analysis
    with open("sae_latent_out/multiple_layers.json", "r") as f:
        results = json.load(f)
    
    layer_features = {
        18: (76, 0.704),
        19: (132, 0.674), 
        20: (132, 0.728)
    }
    
    analysis_results = {}
    
    for layer in [18, 19, 20]:
        print(f"\n{'='*50}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*50}")
        
        # Load SAE for this layer
        sae_id = f"layer_{layer}/{WIDTH_TAG}"
        sae, _, _ = SAE.from_pretrained(SAE_RELEASE, sae_id=sae_id, device=DEVICE)
        
        feature_idx = layer_features[layer][0]
        feature_score = layer_features[layer][1]
        
        print(f"Feature #{feature_idx}, Correlation: {feature_score:.3f}")
        
        # 1. Activation patterns analysis
        print("\n1. Getting feature activations...")
        activations = []
        
        for i in range(0, len(prompts), 8):
            batch = prompts[i:i+8]
            _, cache = model.run_with_cache_with_saes(batch, saes=[sae])
            acts = cache[f"{sae.cfg.hook_name}.hook_sae_output"][:, -1, feature_idx]
            activations.extend(acts.cpu().tolist())
        
        activations = torch.tensor(activations)
        
        # 2. Activation statistics
        harmful_acts = activations[labels == 1]
        harmless_acts = activations[labels == 0]
        
        print(f"Harmful prompts - Mean: {harmful_acts.mean():.4f}, Std: {harmful_acts.std():.4f}")
        print(f"Harmless prompts - Mean: {harmless_acts.mean():.4f}, Std: {harmless_acts.std():.4f}")
        print(f"Separation: {(harmful_acts.mean() - harmless_acts.mean()).abs():.4f}")
        
        # 3. Top activating examples
        print(f"\n2. Top activating examples for Layer {layer} Feature #{feature_idx}:")
        top_indices = torch.topk(activations, 5).indices
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. [{activations[idx]:.3f}] {prompts[idx][:80]}...")
        
        # 4. Feature vector analysis
        feature_vec = sae.W_dec[feature_idx].detach()
        vec_norm = torch.norm(feature_vec).item()
        
        print(f"\n3. Feature vector stats:")
        print(f"  Norm: {vec_norm:.4f}")
        print(f"  Mean: {feature_vec.mean():.6f}")
        print(f"  Std: {feature_vec.std():.6f}")
        
        analysis_results[layer] = {
            "feature_idx": feature_idx,
            "correlation": feature_score,
            "harmful_mean": harmful_acts.mean().item(),
            "harmless_mean": harmless_acts.mean().item(),
            "separation": (harmful_acts.mean() - harmless_acts.mean()).abs().item(),
            "vector_norm": vec_norm,
            "vector_mean": feature_vec.mean().item(),
            "vector_std": feature_vec.std().item(),
            "top_activations": activations[top_indices].tolist(),
            "top_prompts": [prompts[i] for i in top_indices]
        }
    
    # 5. Cross-layer comparisons
    print(f"\n{'='*50}")
    print("CROSS-LAYER COMPARISON")
    print(f"{'='*50}")
    
    # Feature vector similarities
    vec18 = analysis_results[18]["vector_norm"]
    vec19 = analysis_results[19]["vector_norm"] 
    vec20 = analysis_results[20]["vector_norm"]
    
    print(f"\nFeature vector norms:")
    print(f"  Layer 18: {vec18:.4f} (EFFECTIVE)")
    print(f"  Layer 19: {vec19:.4f}")
    print(f"  Layer 20: {vec20:.4f}")
    
    print(f"\nActivation separation (harmful vs harmless):")
    print(f"  Layer 18: {analysis_results[18]['separation']:.4f} (EFFECTIVE)")
    print(f"  Layer 19: {analysis_results[19]['separation']:.4f}")
    print(f"  Layer 20: {analysis_results[20]['separation']:.4f}")
    
    # Save analysis
    with open("layer18_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print("\n✓ Analysis saved to layer18_analysis.json")
    
    return analysis_results

if __name__ == "__main__":
    analyze_layer_differences() 