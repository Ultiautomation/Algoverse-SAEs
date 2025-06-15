import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from collections import Counter
    
def plot_toxicity_score (df):

    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Prepare harmful responses with original index ---

    # Base model harmful responses
    base_harmful_df = df[df['Model_Response_class'] == 'harmful'][['Model_Response_toxicity']].dropna()
    base_harmful_df = base_harmful_df.reset_index().rename(columns={'index': 'PromptIndex'})

    # Instruct model harmful responses
    instruct_harmful_df = df[df['Instruct_Model_Response_class'] == 'harmful'][['Instruct_Model_Response_toxicity']].dropna()
    instruct_harmful_df = instruct_harmful_df.reset_index().rename(columns={'index': 'PromptIndex'})

    # Optional: Sort by toxicity for cleaner visuals
    base_harmful_df = base_harmful_df.sort_values(by='Model_Response_toxicity').reset_index(drop=True)
    instruct_harmful_df = instruct_harmful_df.sort_values(by='Instruct_Model_Response_toxicity').reset_index(drop=True)

    # --- Step 2: Plot setup ---

    # Adjust height dynamically for number of bars, add space for visibility
    height = max(len(base_harmful_df), len(instruct_harmful_df))
    plt.figure(figsize=(14, height / 6))  # Increase figure width and reduce height for better spacing

    #Plot prompt toxicity bars
    prompt_bars = plt.barh(
        y= df['PromptIndex'].astype(str),
        width= df['Prompt_toxicity'],
        color='lightgreen',
        alpha=0.7,
        label='Prompt toxicity score'
    )
    # Plot base model bars (steelblue)
    base_bars = plt.barh(
        y=base_harmful_df['PromptIndex'].astype(str),
        width=base_harmful_df['Model_Response_toxicity'],
        color='steelblue',
        label='Base Model (harmful)',
        height=0.5  # Decrease bar height to increase space between bars
    )

    # Plot instruct model bars (indianred)
    instruct_bars = plt.barh(
        y=instruct_harmful_df['PromptIndex'].astype(str),
        width=instruct_harmful_df['Instruct_Model_Response_toxicity'],
        color='indianred',
        alpha=0.7,
        label='Instruct Model (harmful)',
        height=0.5  # Decrease bar height to increase space between bars
    )

    # --- Step 3: Annotate scores to the right of the bars ---
    for bar in prompt_bars:
        width = bar.get_width()
        # Place the toxicity score outside the bar to the right
        plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', va='center', fontsize=8, color='black')
    for bar in base_bars:
        width = bar.get_width()
        # Place the toxicity score outside the bar to the right
        plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', va='center', fontsize=8, color='black')

    for bar in instruct_bars:
        width = bar.get_width()
        # Place the toxicity score outside the bar to the right
        plt.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', va='center', fontsize=8, color='black')

    # --- Step 4: Adjust x-axis limit dynamically based on data ---

    # Get the maximum toxicity score from both dataframes
    max_base_score = base_harmful_df['Model_Response_toxicity'].max()
    max_instruct_score = instruct_harmful_df['Instruct_Model_Response_toxicity'].max()
    max_prompt_score = df['Prompt_toxicity'].max()

    # Set the max score and apply a margin for better visibility
    max_score = max(max_base_score, max_instruct_score, max_prompt_score)
    plt.xlim(0, max_score * 1.1)  # Add 10% margin for clarity

    # Rotate y-axis labels to avoid overlap
    plt.yticks(rotation=0)

    # --- Step 5: Final chart formatting ---

    plt.xlabel('Toxicity Score')
    plt.ylabel('Prompt Index (from original DataFrame)')
    plt.title('Toxicity Scores of Harmful Responses: Base vs Instruct Model')
    plt.legend()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "toxicity_scores.png"), dpi=300)
    plt.close()  # Close the figure to free memory

def analyse_ig_results (df):
    
    all_scores = []    
    
    for i, ig_list in enumerate(df["Instruct_IG"]):
        if not isinstance(ig_list, list):
            continue  # skip if entry is not a valid list    
        
        scores = []
        for token, score in ig_list:
            if isinstance(score, float) and not math.isnan(score):
                scores.append(score)
        
        all_scores.append({
            "Index": i,
            "Max_Score": max(scores) if scores else None,
            "Min_Score": min(scores) if scores else None,
            "Avg_Score": sum(scores)/len(scores) if scores else None,
            "Num_Tokens": len(scores)
        })
    
    # Create a new DataFrame with these stats
    score_summary_df = pd.DataFrame(all_scores)
    
    # Optional: Merge with the original DataFrame
    ig_results_df = df.join(score_summary_df.set_index("Index"))

    # See top entries
    print(ig_results_df[["Prompt", "Max_Score", "Min_Score","Avg_Score", "Num_Tokens"]].head())
    
    # Make sure the'outputs' directory exists
    os.makedirs ("outputs", exist_ok=True)
    
    # Save the final DataFrame to CSV
    ig_results_df.to_csv("outputs/ig_results_summary.csv", index=False)
    
    print("✅ IG results saved to outputs/ig_results_summary.csv")
    
    # Create output directory if not exists
    output_dir = "outputs/ig_plots"
    os.makedirs (output_dir, exist_ok=True)
            
    # Filter rows with valid IG data
    valid_rows = ig_results_df[ig_results_df["Instruct_IG"].apply(lambda x: isinstance(x, list))]
    
    # Batch size
    BATCH_SIZE = 20
    num_rows = len(valid_rows)
    print(f"number of ig results found: {num_rows}")
    num_batches = math.ceil(num_rows / BATCH_SIZE)
    print(f"generating plots in {num_batches} batches")
    
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_rows)
    
        batch = valid_rows.iloc[start_idx:end_idx]
    
        # Set up subplots
        fig, axs = plt.subplots(nrows=len(batch), figsize=(12, len(batch) * 3), constrained_layout=True)
    
        if len(batch) == 1:
            axs = [axs]  # Ensure axs is iterable
    
        # Plot each row in batch
        for idx, (i, row) in enumerate(batch.iterrows()):
            ig_data = row["Instruct_IG"]
    
            tokens = []    
            scores = []
            for token, score in ig_data:
                if isinstance(score, float) and not math.isnan(score):
                    tokens.append(token)
                    scores.append(score)
    
            axs[idx].bar(tokens, scores, color="skyblue")
            axs[idx].set_title(f"Prompt {i}", fontsize=12)
            axs[idx].tick_params(axis='x', labelrotation=45)
            axs[idx].set_ylabel("Attribution Score")
            axs[idx].set_xlabel("Tokens")
    
        # Save figure
        save_path = os.path.join(output_dir, f"ig_scores_batch_{batch_num + 1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f" Saved batch {batch_num +1} to {save_path}")
        
def find_most_common_refusal_term (df)
    output_dir = "outputs/plots"
    os.makedirs (output_dir, exist_ok=True)
    refusal_counts = sampled_df['Refusal_outputs'].dropna().explode().value_counts()
    refusal_counts.plot(kind='bar', figsize=(12, 4), title='Most Common Refusal Phrases')
    plt.tight_layout()
    plt.savefig(os.join(output_dir, "common_refusal_terms.png"), dpi=300)
    plt.close()