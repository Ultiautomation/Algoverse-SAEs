import pandas as pd
import matplotlib.pyplot as plt
import os

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


