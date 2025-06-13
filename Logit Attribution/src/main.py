from data_loader import load_sampled_dataframe
from model_loader import get_tokenizer, load_model, unload_model
from attribution.visualize import plot_toxicity_score
import sys

def main():
    # Load toxicity scored data (CSV data)
    scored_df = load_sampled_dataframe("/root/Algoverse-SAEs/Logit Attribution/data/gemma_responses_toxicity_scoring_with_prompt_scoring.csv")
    scored_df = scored_df.reset_index().rename(columns={'index': 'PromptIndex'})
    print(f"Columns in the scored dataframe: {scored_df.columns}")

    # Visualise the toxicity scores for sampling purposes
    plot_toxicity_score (scored_df)

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