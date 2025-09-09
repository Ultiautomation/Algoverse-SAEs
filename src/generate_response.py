from data_loader import load_sampled_dataframe
from model_loader import get_tokenizer, load_model, unload_model
from intergrated_gradients import run_ig, get_baseline, run_layerwise_conductance
from visualize import plot_toxicity_score, analyse_ig_results, find_most_common_refusal_term
from extract_data import extract_refusal_phrases
from config import load_dotenv
from huggingface_hub import login
import sys
import torch
from tqdm import tqdm
import pandas as pd
import time
import os
import pickle
from dotenv import load_dotenv
load_dotenv()
hugging_face_token = os.getenv('hugging_face_token')

def main():
    model_name = "tomg-group-umd/zephyr-llama3-8b-sft-refusal-n-contrast-multiple-tokens"    
    tokenizer = get_tokenizer(model_name)      
    
    model = load_model(model_name)
    print("Tuned model loaded.")
    
    # Load existing file
    save_path = "outputs/sampled.csv"
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        prompts = existing_df["Prompt"].tolist()
        # Check if 'Instruct_Model_Response' exists
        if "Instruct_Model_Response_New" in existing_df.columns:
            responses = existing_df["Instruct_Model_Response_New"].tolist()
        else:
            responses = [""] * len(prompts)
            existing_df["Instruct_Model_Response_New"] = responses
    else:
        raise FileNotFoundError("CSV with prompts not found. Please create or specify the file.")
    
    # Resume inference from first empty response
    start_index = next((i for i, resp in enumerate(responses) if not isinstance(resp, str) or not resp.strip()), len(prompts))
    
    for i in tqdm(range(start_index, len(prompts)), desc="Generating responses"):
        prompt = prompts[i]
        print(f"\n Prompt {i}:\n{prompt}\n")
    
        try:
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature= 0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_output[len(prompt):].strip()
            elapsed = time.time() - start_time
    
            print(f"🕒 Time taken: {elapsed:.2f}s")
            print(f" Response:\n{response}\n{'-'*80}")
    
        except Exception as e:    
            print(f" Error at index {i}: {e}")
            response = "[ERROR]"
    
        # Update and save
        existing_df.at[i, "Instruct_Model_Response_New"] = response    
        existing_df.to_csv(save_path, index=False)
    
    print("✅ All instruct responses saved and merged into CSV.")

if __name__ == "__main__":
    main()