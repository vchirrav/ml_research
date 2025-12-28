import json
import torch
import warnings
import multiprocessing
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Hardware & Warnings setup
warnings.filterwarnings("ignore")


def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_chat_template(example):
    messages = example["messages"]
    conversation = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        conversation += f"<|{role}|>\n{content}\n"
    conversation += "<|endoftext|>"
    return {"text": conversation}


# --- PROTECT THE EXECUTION ---
if __name__ == "__main__":
    # Windows-specific safety
    multiprocessing.freeze_support()

    print(f"--- D.Eng Research Environment ---")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_120 Native Active)")

    # 2. Data Loading
    dataset_path = "resume_training_dataset.jsonl"
    raw_data = load_jsonl_dataset(dataset_path)
    formatted_data = [format_chat_template(ex) for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 3. Model & Quantization
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Changed from float16
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,  # Changed from float16
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

    # 5. SFT Config (Updated for 2025/Blackwell)
    training_args = SFTConfig(
        output_dir="./cybersecurity_expert_model",
        max_length=2048,  # Using max_length for 2025 TRL compatibility
        dataset_text_field="text",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        load_best_model_at_end=True,
        report_to="none",
        dataset_num_proc=1,  # Explicitly keep at 1 for Windows stability
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )

    print("\nTrainer initialized. Starting Blackwell-optimized training!")
    trainer.train()

    # 7. Save Model
    trainer.save_model("./cybersecurity_expert_model_final")
    print("Training complete. Model saved.")
