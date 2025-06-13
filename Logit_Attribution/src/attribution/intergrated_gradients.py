from captum.attr import IntegratedGradients
import torch
import pandas as pd
import time
import pickle
import os
import gc

from model_loader import get_tokenizer, load_model, unload_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()

def get_baseline(input_ids, model, tokenizer, strategy="pad"):
        """Generate a baseline tensor for Integrated Gradients."""
        embedding_layer = model.get_input_embeddings()
        
        if strategy == "zero":
            return torch.zeros_like(embedding_layer(input_ids))

        elif strategy == "pad":
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                raise ValueError("Tokenizer does not have a pad token.")
            pad_ids = torch.full_like(input_ids, pad_token_id)
            return embedding_layer(pad_ids)

        else:
            raise ValueError(f"Unknown baseline strategy: {strategy}")
        
def run_ig(prompt, model, tokenizer, refusal_terms, baseline_strategy="pad", n_steps=20):
    #  """
    #     Run Integrated Gradients on a prompt for a list of refusal terms.

    #     Parameters:
    #     - prompt (str): The input prompt.
    #     - model: The language model (must support inputs_embeds).
    #     - tokenizer: Hugging Face tokenizer corresponding to the model.
    #     - refusal_terms (List[str]): List of refusal phrases to attribute against.
    #     - baseline_strategy (str): "pad" or "zero" baseline.
    #     - n_steps (int): Number of interpolation steps for IG.

    #     Returns:
    #     - List[Tuple[token, attribution_score]]
    # """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize and embed input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    embeddings = model.get_input_embeddings()(input_ids)
    baseline = get_baseline(input_ids, model, tokenizer, strategy=baseline_strategy).to(device)

    total_attribution = torch.zeros_like(embeddings)

    # Loop over each refusal phrase
    for phrase in refusal_terms:
        phrase = phrase.strip()
        if not phrase:
            continue

        target_tokens = tokenizer.tokenize(phrase)
        token_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        if len(token_ids) == 0:
            print(f"Skipping unknown phrase: {phrase}")
            continue

        # Loop over each token ID
        for token_id in token_ids:
            def forward_func(embeds):
                logits = model(inputs_embeds=embeds, attention_mask=attention_mask).logits
                return logits[:, -1, token_id]

            ig = IntegratedGradients(forward_func)
            try:
                baseline = get_baseline(input_ids, model, tokenizer, strategy="pad")
                attributions, delta = ig.attribute(
                    embeddings,
                    baselines = baseline,
                    target=None,
                    n_steps=20,
                    return_convergence_delta=True
                )
                total_attribution += attributions
            except Exception as e:
                print(f"IG failed for token_id {token_id}: {e}")
                continue

    token_scores = total_attribution.sum(dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return list(zip(tokens, token_scores.tolist()))

def run_layerwise_ig(prompt, model, tokenizer, refusal_terms, baseline_strategy="pad", n_steps=20):
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    layerwise_attributions = {}
    aggregate_token_scores = {}

    for layer_idx, layer_hidden in enumerate(hidden_states):
        embeddings = layer_hidden.requires_grad_().to(device)
        baseline = get_baseline(input_ids, model, tokenizer, strategy=baseline_strategy).to(device)

        layer_token_attrs = {}

        for phrase in refusal_terms:
            phrase = phrase.strip()
            if not phrase:
                continue

            target_tokens = tokenizer.tokenize(phrase)
            token_ids = tokenizer.convert_tokens_to_ids(target_tokens)

            if len(token_ids) == 0:
                print(f"Skipping unknown phrase: {phrase}")
                continue

            for token_id in token_ids:
                token_str = tokenizer.convert_ids_to_tokens([token_id])[0]

                def wrapped_forward(embeds):
                    return forward_func(embeds, attention_mask, model, token_id)

                lig = LayerIntegratedGradients(wrapped_forward, model.get_input_embeddings())
                try:
                    attributions, _ = lig.attribute(
                        inputs=embeddings,
                        baselines=baseline,
                        n_steps=n_steps,
                        return_convergence_delta=True
                    )
                    attributions = attributions.detach().cpu()
                    layer_token_attrs[token_str] = attributions

                    # Aggregate across layers
                    token_score = attributions.sum(dim=-1).squeeze(0)
                    if token_str not in aggregate_token_scores:
                        aggregate_token_scores[token_str] = torch.zeros_like(token_score)
                    aggregate_token_scores[token_str] += token_score

                except Exception as e:
                    print(f"Layer {layer_idx} - Token '{token_str}' IG failed: {e}")

        # Postprocess per-layer
        token_scores_dict = {}
        for token_str, attr_tensor in layer_token_attrs.items():
            token_score = attr_tensor.sum(dim=-1).squeeze(0)
            token_scores_dict[token_str] = list(zip(tokens, token_score.tolist()))

        layerwise_attributions[f"layer_{layer_idx}"] = token_scores_dict

    # Final aggregate attribution
    aggregate_token_scores_readable = {
        token_str: list(zip(tokens, score.tolist()))
        for token_str, score in aggregate_token_scores.items()
    }

    return {
        "layerwise": layerwise_attributions,
        "aggregate": aggregate_token_scores_readable
    }
