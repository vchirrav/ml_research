import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "./cybersecurity_expert_model_final"
save_path = "./merged_cyber_model"

# 1. Load the base model (use float16 for better compatibility)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    device_map="cpu"
)

# 2. Load the adapter and merge
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# 3. Save the result
merged_model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(save_path)

print(f"Model successfully merged to {save_path}")