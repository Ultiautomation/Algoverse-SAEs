from data_loader import load_sampled_dataframe
from model_loader import get_tokenizer, load_model, unload_model
from attribution.visualize import plot_toxicity_score
from extract_data import extract_refusal_phrases
import sys

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


    # Set model names
    base_model_name = "google/gemma-2-2b"
    tuned_model_name = "google/gemma-2-2b-it"
    sys.exit()
    # Load tokenizer once
    tokenizer = get_tokenizer(base_model_name)

    # Load and unload base model (as an example)
    tuned_model = load_model(tuned_model_name)
    print("Base model loaded.")
    
    # [Do something with model here...]

    # Clean up
    unload_model(tuned_model)
    print("Model unloaded from memory.")

if __name__ == "__main__":
    main()