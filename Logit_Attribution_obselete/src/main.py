from data_loader import load_sampled_dataframe
from model_loader import get_tokenizer, load_model, load_model_with_hidden_states, unload_model
from attribution.intergrated_gradients import run_ig, get_baseline
from attribution.visualize import plot_toxicity_score
from extract_data import extract_refusal_phrases
from config import load_dotenv
from huggingface_hub import login
import sys
import pandas as pd
import time
import os
import pickle
from dotenv import load_dotenv
load_dotenv()
hugging_face_token = os.getenv('hugging_face_token')

def main():
    # Load toxicity scored data (CSV data)
    scored_df = load_sampled_dataframe("/root/Algoverse-SAEs/Logit Attribution/data/gemma_responses_toxicity_scoring_with_prompt_scoring.csv")
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
    
    # Ensure you are logged in to huggingface
    login(token= hugging_face_token)

    # Set model names
    base_model_name = "google/gemma-2-2b"
    tuned_model_name = "google/gemma-2-2b-it"
    
    # Load tokenizer once
    tokenizer = get_tokenizer(base_model_name)

    # Settings to intitiate IG
    SAVE_EVERY = 10
    INTERMEDIATE_SAVE_PATH = "outputs/ig_partial_results.pkl"

    # Restore partial results
    records = []
    start_index = 0
    if os.path.exists(INTERMEDIATE_SAVE_PATH):
        with open(INTERMEDIATE_SAVE_PATH, "rb") as f:
            records = pickle.load(f)
        start_index = len(records)
        print(f"Resuming from index {start_index}")

    # Run attribution
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
            print("tuned model loaded.")
            instruct_ig = run_ig(prompt, instruct_model, tokenizer, refusal_terms, baseline_strategy="pad", n_steps=20)
            unload_model(instruct_model)
            print("tuned model unloaded from memory.")
            elapsed = time.time() - start_time

            records.append({
                'Prompt': prompt,
                'Prompt_toxicity': row['Prompt_toxicity'],
                'Instruct_Response': row['Instruct_Model_Response'],
                'Instruct_IG': instruct_ig,
                'Time_Taken': elapsed
            })

            if (i + 1) % SAVE_EVERY == 0:
                with open(INTERMEDIATE_SAVE_PATH, "wb") as f:
                    pickle.dump(records, f)
                print(f"✅ Saved progress at {i + 1} prompts.")

        except Exception as e:
            print(f"❌ Error on prompt {i}: {e}")
            continue

    # Final save
    new_df = pd.DataFrame(records)
    new_df.to_pickle("ig_full_results.pkl")

    ## RUN IG LAYERWISE
    # Settings for layerwise IG
    SAVE_EVERY = 10
    LAYERWISE_SAVE_PATH = "outputs/layerwise_ig_partial_results.pkl"
    FINAL_SAVE_PATH = "outputs/layerwise_ig_full_results.pkl"

    # Restore partial results
    layerwise_records = []
    start_index = 0
    if os.path.exists(LAYERWISE_SAVE_PATH):
        with open(LAYERWISE_SAVE_PATH, "rb") as f:
            layerwise_records = pickle.load(f)
        start_index = len(layerwise_records)
        print(f"Resuming layerwise IG from index {start_index}")

    # Run layerwise IG attribution
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

            layerwise_ig_result = run_layerwise_ig(
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

            layerwise_records.append({
                'Prompt': prompt,
                'Prompt_toxicity': row['Prompt_toxicity'],
                'Instruct_Response': row['Instruct_Model_Response'],
                'Layerwise_IG': layerwise_ig_result,
                'Time_Taken': elapsed
            })

            if (i + 1) % SAVE_EVERY == 0:
                with open(LAYERWISE_SAVE_PATH, "wb") as f:
                    pickle.dump(layerwise_records, f)
                print(f"✅ Layerwise IG saved progress at prompt {i + 1}")

        except Exception as e:
            print(f"❌ Error on prompt {i}: {e}")
            continue

    # Final save
    layerwise_df = pd.DataFrame(layerwise_records)
    layerwise_df.to_pickle(FINAL_SAVE_PATH)
    print(f"\n✅ Layerwise IG completed. Results saved to {FINAL_SAVE_PATH}")


if __name__ == "__main__":
    main()