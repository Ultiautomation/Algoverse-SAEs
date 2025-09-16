import os
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from config import load_dotenv
from huggingface_hub import login
load_dotenv()
hugging_face_token = os.getenv('hugging_face_token')

def main():

    # === CONFIG ===
    CSV_PATH = "outputs/sampled_new.csv"  
    OUTPUT_JSONL = "outputs/contrastive_dataset_llama.jsonl"
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    SAVE_EVERY = 5
    
    # === Load Model and Tokenizer ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,    
        max_new_tokens =100,
        do_sample= True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # === Prompt Formatting for LLaMA-3 ===
    def format_prompt_llama3(prompt):
        return (
            "<|begin_of_text|><|start_hea    der_id|>system<|end_header_id|>\n"
            "You are a helpful, honest assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}<|eot_id|><|start_hea    der_id    |    >assistant<|end_header_id|>\    n"
        )
    
    # === Resume Capability ===
    existing_questions = set()
    if os.path.exists(OUTPUT_JSONL):
        with open (OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_questions.add(obj["quest    ion"])
                except json.JSONDecodeError:
                    continue
    
    # === Load Data ===
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Prompt", "Instruct_Model_Response_New", "category"])
    
    # === Process Rows ===
    buffer = []
    
    for i, row in df.iterrows():
        prompt = str(row["Prompt"]).strip().strip('"')
        print("prompt:", prompt)
        refusal = str(row["Instruct_Model_Response_New"]).strip()
        print("Option A:", refusal)
        category = str(row["category"]).strip()    
        print("category:",category)
    
        base_question = f"{prompt}\n\nChoices:\n(A) {refusal} [{category}]    .    "
        if base_question in existing_questions:
            continue
    
        # === Generate Response with LLaMA Chat Format ===
        llama_prompt = format_prompt_llama3(prompt)
        try:
            output = generator(llama_prompt, return_full_text=False)[0]['generated_text']
            # Clean up generated response
            response = output.strip().split("\n")[0].rstrip(".") + ". [respond]"
            print("response:", response)
        except Exception as e:
            print(f"❌ Generation failed for prompt at row {i}: {e}")
            response = "[GENERATION_FAILED] [respond]"
    
        # === Build final contrastive question ===
        full_question = (
            f"{prompt}\n\nChoices:\n"
            f"(A) {refusal} [{category}].\n"
            f"(B) {response}"
        )
    
        entry = {
            "question": full_question,
            "answer_matching_behavior": "(A)",
            "answer_not_matching_behavior": "(B)"
        }
    
        buffer.append(entry)
    
        # Save every N
        if len(buffer) >= SAVE_EVERY:
            with open(OUTPUT_JSONL, 'a', encoding='utf-8') as out:
                for item in buffer:
                    out.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✅ Saved {len(buffer)} entries (up to row {i})")
            buffer.clear()
    
    # Final save
    if buffer:
        with open(OUTPUT_JSONL, 'a', encoding='utf-8') as out:
            for item in buffer:
                out.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ Final save of {len(buffer)} entries")
    
    print("🎉 DONE! All contrastive pairs generated and saved.")

if __name__ == "__main__":
    main()
