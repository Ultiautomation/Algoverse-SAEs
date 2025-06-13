import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def load_model(model_name: str, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        # device_map="auto"  # Uncomment if using multiple GPUs
    )
    return model

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()