import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaModel
import gc

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def load_model(model_name: str, dtype=torch.float16):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=dtype,
    #     output_hidden_states=True
    #     # device_map="auto"  # Uncomment if using multiple GPUs
    # )
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model

def load_model_with_hidden_states(model_name: str, dtype=torch.float16):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=dtype,
    #     output_hidden_states=True
    #     # device_map="auto"  # Uncomment if using multiple GPUs
    # )
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        output_hidden_states=True
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model

def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()