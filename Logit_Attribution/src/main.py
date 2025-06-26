from data_loader import load_sampled_dataframe
from model_loader import get_tokenizer, load_model, unload_model
from intergrated_gradients import run_ig, get_baseline
from visualize import plot_toxicity_score, analyse_ig_results, find_most_common_refusal_term
from extract_data import extract_refusal_phrases
from config import load_dotenv
from huggingface_hub import login
import sys
import torch
import pandas as pd
import time
import os
import pickle
from dotenv import load_dotenv
load_dotenv()
hugging_face_token = os.getenv('hugging_face_token')

def main():
    # Set output file path
    sampled_path = "outputs/sampled.csv"
    
    # Check if sampled.csv exists
    if os.path.exists(sampled_path):
        print("✅ Found existing 'sampled.csv', loading it...")
        sampled_df = pd.read_csv(sampled_path)
        pd.set_option('display.max_colwidth', None)
        print(sampled_df.head())
        sampled_path1 = "outputs/plots/common_refusal_terms.png"
        if os.path.exists(sampled_path1):
            print("✅ Found existing 'common_refusal_terms.png', no need to generate...")
    else:
        print("⚙️ No 'sampled.csv' found. Running sampling...")
        # Load toxicity scored data (CSV data)
        if "Refusal_outputs" not in sampled_df.columns:
            # Extract refusal phrases and insert a new column to prepare for IG
            refusal_terms = [
            "i'm sorry", "i am sorry", "i cannot", "i can't", "i am unable", 
            "as an ai", "i do not", "my apologies", "i won't", "i shouldn’t", 
            "unfortunately", "it is not appropriate", "i don't have the ability", 
            "i'm not able", "i do not have", "do not follow", "is a serious crime", 
            "illegal", "harmful", "unethical", "should not be"]
        
            sampled_df["Refusal_outputs"] = sampled_df["Instruct_Model_Response"].apply(
            lambda x: extract_refusal_phrases(x, refusal_terms)
            )
        
            refusal_is_na = sampled_df['Refusal_outputs'].isna().any()
        
            print(f"Prompts have a corresponding NaN value for refusal phrase: {refusal_is_na}")
        else: 
            print("'Refusal_outputs' column already exists. Skipping extraction step.")
            find_most_common_refusal_term(sampled_df)
            print("most_common_refusal_terms_plot generated")
        scored_df = load_sampled_dataframe("/workspace/data/gemma_responses_toxicity_scoring_with_prompt_scoring.csv")
        scored_df = scored_df.reset_index().rename(columns={'index': 'PromptIndex'})
        print(f"Columns in the scored dataframe: {scored_df.columns}")
    
        # Visualise the toxicity scores for sampling purposes
        plot_toxicity_score (scored_df)
    
        #Sampling
        # 1. Filter prompts above average toxicity
        avg_toxicity = scored_df['Prompt_toxicity'].mean()
        filtered_df = scored_df[scored_df['Prompt_toxicity'] > avg_toxicity]
    
        # 2. Determine sample size and sample
        sample_size = len(filtered_df)
        print(f"Sample size: {sample_size} out of {len(scored_df)}")
    
        sampled_df = filtered_df.copy()
        pd.set_option('display.max_colwidth', None)
        print (sampled_df)

        # ✅ Save sampled dataframe to file
        os.makedirs("outputs", exist_ok=True)  # ensure the directory exists
        sampled_df.to_csv(sampled_path, index=False)
        print(f"📁 Sampled data saved to: {sampled_path}")
        
    # if "Refusal_outputs" not in sampled_df.columns:
    #     # Extract refusal phrases and insert a new column to prepare for IG
    #     refusal_terms = [
    #     "i'm sorry", "i am sorry", "i cannot", "i can't", "i am unable", 
    #     "as an ai", "i do not", "my apologies", "i won't", "i shouldn’t", 
    #     "unfortunately", "it is not appropriate", "i don't have the ability", 
    #     "i'm not able", "i do not have", "do not follow", "is a serious crime", 
    #     "illegal", "harmful", "unethical", "should not be"]
    
    #     sampled_df["Refusal_outputs"] = sampled_df["Instruct_Model_Response"].apply(
    #     lambda x: extract_refusal_phrases(x, refusal_terms)
    #     )
    
    #     refusal_is_na = sampled_df['Refusal_outputs'].isna().any()
    
    #     print(f"Prompts have a corresponding NaN value for refusal phrase: {refusal_is_na}")
    # else: 
    #     print("'Refusal_outputs' column already exists. Skipping extraction step.")
    
    # Ensure you are logged in to huggingface
    login(token= hugging_face_token)

    # Set model names
    base_model_name = "google/gemma-2-2b"
    tuned_model_name = "google/gemma-2-2b-it"
    
    # Load tokenizer once
    tokenizer = get_tokenizer(base_model_name)

    # Uncomment the below code block to run IG for final output
    --------------------------------------------------------------
    # # Settings to intitiate IG 
    # SAVE_EVERY = 10
    # INTERMEDIATE_SAVE_PATH = "outputs/ig_partial_results.pkl"

    # # Restore partial results
    # records = []
    # start_index = 0
    # if os.path.exists(INTERMEDIATE_SAVE_PATH):
    #     with open(INTERMEDIATE_SAVE_PATH, "rb") as f:
    #         records = pickle.load(f)
    #     start_index = len(records)
    #     print(f"Resuming from index {start_index}")

    # # Run attribution
    # for i, (_, row) in enumerate(sampled_df.iloc[start_index:].iterrows(), start=start_index):
    #     prompt = row['Prompt']
    #     print(f"\n[Prompt {i}] {prompt}")

    #     refusal_terms = row.get("Refusal_outputs", "").split("; ")
    #     refusal_terms = [term.strip() for term in refusal_terms if term.strip()]

    #     if not refusal_terms:
    #         print("No refusal terms found; skipping.")
    #         continue

    #     try:
    #         start_time = time.time()
    #         instruct_model = load_model(tuned_model_name)
    #         print("tuned model loaded.")
    #         instruct_ig = run_ig(prompt, instruct_model, tokenizer, refusal_terms, baseline_strategy="pad", n_steps=20)
    #         unload_model(instruct_model)
    #         print("tuned model unloaded from memory.")
    #         elapsed = time.time() - start_time

    #         records.append({
    #             'Prompt': prompt,
    #             'Prompt_toxicity': row['Prompt_toxicity'],
    #             'Instruct_Response': row['Instruct_Model_Response'],
    #             'Instruct_IG': instruct_ig,
    #             'Time_Taken': elapsed
    #         })

    #         if (i + 1) % SAVE_EVERY == 0:
    #             with open(INTERMEDIATE_SAVE_PATH, "wb") as f:
    #                 pickle.dump(records, f)
    #             print(f"✅ Saved progress at {i + 1} prompts.")

    #     except Exception as e:
    #         print(f"❌ Error on prompt {i}: {e}")
    #         continue

    # # Final save
    # new_df = pd.DataFrame(records)
    # new_df.to_pickle("ig_full_results.pkl")

    # #Analyse
    # ig_results_df = pd.read_pickle("ig_full_results.pkl")
    # analyse_ig_results(ig_results_df)
    -------------------------------------------------------------------------------------------

    #layerwise Run_IG
    # Settings
    SAVE_EVERY = 10
    LAYERWISE_SAVE_PATH = "outputs/layerwise_ig_partial_results.pkl"
    FINAL_SAVE_PATH = "outputs/layerwise_ig_full_results.pkl"
    DATAFRAME_SAVE_PATH = "outputs/layerwise_outputs.csv"
    
    # Restore partial results
    layerwise_records = []
    start_index = 0
    if os.path.exists(LAYERWISE_SAVE_PATH):
        with open(LAYERWISE_SAVE_PATH, "rb") as f:
            layerwise_records = pickle.load(f)
        start_index = len(layerwise_records)
        print(f"Resuming layerwise IG from index {start_index}")
    
    # Main loop
    for i, (_, row) in enumerate(sampled_df.iloc[start_index:].iterrows(), start=start_index):
        prompt = row['Prompt']
        print(f"\n[Prompt {i}] {prompt}")
    
        try:
            start_time = time.time()
    
            # Load model
            instruct_model = load_model(tuned_model_name)
            instruct_model.eval().cuda()
            device = next(instruct_model.parameters()).device
    
            # Prepare inputs
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = instruct_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
    
            lm_head = instruct_model.lm_head
            layerwise_data = {}
    
            for layer_idx, layer_hidden in enumerate(hidden_states):
                print(f"\n🔍 Processing Layer {layer_idx}...")
    
                # Project to vocab space and decode
                logits = lm_head(layer_hidden)  # [1, seq_len, vocab]
                pred_ids = torch.argmax(logits, dim=-1)  # [1, seq_len]
                intermediate_response = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
    
                # Save decoded response to dataframe
                sampled_df.at[i, f"refusal_output_layer_{layer_idx}"] = intermediate_response
                print(f"[Layer {layer_idx}] Intermediate Response: {intermediate_response}")
    
                refusal_terms = [intermediate_response.strip()] if intermediate_response.strip() else []
    
                if not refusal_terms:
                    print(f"⚠️ No valid refusal terms at layer {layer_idx}")
                    continue
    
                try:
                    ig_result = run_ig(
                        prompt,
                        instruct_model,
                        tokenizer,
                        refusal_terms,
                        baseline_strategy="pad",
                        n_steps=20
                    )
                except Exception as ig_err:
                    print(f"❌ IG failed at layer {layer_idx}: {ig_err}")
                    ig_result = []
    
                # Save IG result
                layerwise_data[f"ig_layer_{layer_idx}"] = ig_result
    
            # Unload model
            unload_model(instruct_model)
            print("✅ Model unloaded.")
    
            elapsed = time.time() - start_time
    
            # Store all results
            layerwise_record = {
                'Prompt': prompt,
                'Prompt_toxicity': row.get('Prompt_toxicity', None),
                'Instruct_Response': row.get('Instruct_Model_Response', None),
                'Layerwise_IG_Results': layerwise_data,
                'Time_Taken': elapsed
            }
            layerwise_records.append(layerwise_record)
    
            # Periodic save
            if (i + 1) % SAVE_EVERY == 0:
                with open(LAYERWISE_SAVE_PATH, "wb") as f:
                    pickle.dump(layerwise_records, f)
                sampled_df.to_csv(DATAFRAME_SAVE_PATH, index=False)
                print(f"💾 Saved progress and DataFrame at prompt {i + 1}")
    
        except Exception as e:
            print(f"❌ Error on prompt {i}: {e}")
            continue
    
    # Final save
    with open(FINAL_SAVE_PATH, "wb") as f:
        pickle.dump(layerwise_records, f)
    sampled_df.to_csv(DATAFRAME_SAVE_PATH, index=False)
    print("✅ All results and DataFrame saved.")

    ----------------------------------------------------------------------------------------
    # First code for layerwise IG function  - refer intergrated_gradients.py for the function
    # ## RUN IG LAYERWISE
    # # Settings for layerwise IG
    # SAVE_EVERY = 10
    # LAYERWISE_SAVE_PATH = "outputs/layerwise_ig_partial_results.pkl"
    # FINAL_SAVE_PATH = "outputs/layerwise_ig_full_results.pkl"

    # # Restore partial results
    # layerwise_records = []
    # start_index = 0
    # if os.path.exists(LAYERWISE_SAVE_PATH):
    #     with open(LAYERWISE_SAVE_PATH, "rb") as f:
    #         layerwise_records = pickle.load(f)
    #     start_index = len(layerwise_records)
    #     print(f"Resuming layerwise IG from index {start_index}")

    # # Run layerwise IG attribution
    # for i, (_, row) in enumerate(sampled_df.iloc[start_index:].iterrows(), start=start_index):
    #     prompt = row['Prompt']
    #     print(f"\n[Prompt {i}] {prompt}")

    #     refusal_terms = row.get("Refusal_outputs", "").split("; ")
    #     refusal_terms = [term.strip() for term in refusal_terms if term.strip()]

    #     if not refusal_terms:
    #         print("No refusal terms found; skipping.")
    #         continue

    #     try:
    #         start_time = time.time()
    #         instruct_model = load_model(tuned_model_name)
    #         print("Tuned model loaded.")

    #         layerwise_ig_result = run_layerwise_ig(
    #             prompt,
    #             instruct_model,
    #             tokenizer,
    #             refusal_terms,
    #             baseline_strategy="pad",
    #             n_steps=20
    #         )

    #         unload_model(instruct_model)
    #         print("Tuned model unloaded from memory.")
    #         elapsed = time.time() - start_time

    #         layerwise_records.append({
    #             'Prompt': prompt,
    #             'Prompt_toxicity': row['Prompt_toxicity'],
    #             'Instruct_Response': row['Instruct_Model_Response'],
    #             'Layerwise_IG': layerwise_ig_result,
    #             'Time_Taken': elapsed
    #         })

    #         if (i + 1) % SAVE_EVERY == 0:
    #             with open(LAYERWISE_SAVE_PATH, "wb") as f:
    #                 pickle.dump(layerwise_records, f)
    #             print(f"✅ Layerwise IG saved progress at prompt {i + 1}")

    #     except Exception as e:
    #         print(f"❌ Error on prompt {i}: {e}")
    #         continue

    # # Final save
    # layerwise_df = pd.DataFrame(layerwise_records)
    # layerwise_df.to_pickle(FINAL_SAVE_PATH)
    # print(f"\n✅ Layerwise IG completed. Results saved to {FINAL_SAVE_PATH}")

    ## RUN LAYERWISE CONSDUCTANCE
    # Settings for layerwise conductance
    SAVE_EVERY = 10
    LAYERWISE_SAVE_PATH = "outputs/layerwise_cond_partial_results.pkl"
    FINAL_SAVE_PATH = "outputs/layerwise_cond_full_results.pkl"

    # Restore partial results
    layerwise_cond_records = []
    start_index = 0
    if os.path.exists(LAYERWISE_COND_SAVE_PATH):
        with open(LAYERWISE_COND_SAVE_PATH, "rb") as f:
            layerwise_cond_records = pickle.load(f)
        start_index = len(layerwise_cond_records)
        print(f"Resuming layerwise IG from index {start_index}")

    # Run layerwise conductance 
    for i, (_, row) in enumerate(sampled_df.iloc[start_index:].iterrows(), start=start_index):
        prompt = row['Prompt']
        print(f"\n[Prompt {i}] {prompt}")

        refusal_terms = row.get("Refusal_outputs", "").split("; ")
        refusal_terms = [term.strip() for term in refusal_terms if term.strip()]

        if not refusal_terms:
            print("No refusal terms found; skipping.")
            continue

        try:
            start_time = time.time()
            instruct_model = load_model(tuned_model_name)
            print("Tuned model loaded.")

            layerwise_cond_result = run_layerwise_conductance(
                prompt,
                instruct_model,
                tokenizer,
                refusal_terms,
                baseline_strategy="pad",
                n_steps=20
            )

            unload_model(instruct_model)
            print("Tuned model unloaded from memory.")
            elapsed = time.time() - start_time

            layerwise_cond_records.append({
                'Prompt': prompt,
                'Prompt_toxicity': row['Prompt_toxicity'],
                'Instruct_Response': row['Instruct_Model_Response'],
                'Layerwise_Conductance': layerwise_cond_result,
                'Time_Taken': elapsed
            })

            if (i + 1) % SAVE_EVERY == 0:
                with open(LAYERWISE_SAVE_PATH, "wb") as f:
                    pickle.dump(layerwise_cond_records, f)
                print(f"✅ Layerwise Conductance saved progress at prompt {i + 1}")

        except Exception as e:
            print(f"❌ Error on prompt {i}: {e}")
            continue

    # Final save
    layerwise_cond_df = pd.DataFrame(layerwise_cond_records)
    layerwise_cond_df.to_pickle(FINAL_SAVE_PATH)
    print(f"\n✅ Layerwise conductance completed. Results saved to {FINAL_SAVE_PATH}")

if __name__ == "__main__":
    main()
