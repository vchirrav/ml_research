## Project Overview
This project fine-tunes the TinyLlama-1.1B-Chat-v1.0 model for cybersecurity expertise using LoRA (Low-Rank Adaptation) adapters. The fine-tuned model is then converted to GGUF format and deployed using Ollama.

## Environment Setup

### Hardware & Software
- **GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120 Native Active - Blackwell architecture)
- **PyTorch**: 2.10.0.dev20251211+cu128
- **Python Environment**: Managed with `uv`
- **Platform**: Windows

### Key Technologies
- **Transformers**: HuggingFace library for model loading and training
- **PEFT**: Parameter-Efficient Fine-Tuning with LoRA
- **TRL**: Transformer Reinforcement Learning with SFTTrainer
- **BitsAndBytes**: 4-bit quantization for efficient training
- **llama.cpp**: For GGUF conversion
- **Ollama**: For local model deployment

## Training Pipeline

### 1. Fine-Tuning with LoRA Adapters

**Script**: [Vis_local_finetune_transformers.py](Vis_local_finetune_transformers.py)

**Command**:
```bash
uv run .\Vis_local_finetune_transformers.py
```

**Configuration**:
- **Base Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Quantization**: 4-bit NF4 with bfloat16 compute dtype
- **LoRA Parameters**:
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Parameters**:
  - Epochs: 3
  - Batch size: 1 per device
  - Gradient accumulation steps: 4
  - Learning rate: 2e-4
  - Max sequence length: 2048
  - Optimizer: paged_adamw_8bit
  - Scheduler: cosine
  - Precision: bfloat16 (optimized for Blackwell GPU)

**Dataset**: [resume_training_dataset.jsonl](resume_training_dataset.jsonl)
- Training samples: 20
- Evaluation samples: 3
- Format: Chat template with role-based messages

**Training Results**:
- Final evaluation loss: 1.5549
- Mean token accuracy: 64.57%
- Training time: ~24 seconds
- Output: `./cybersecurity_expert_model_final/`

### 2. Merging LoRA Adapters with Base Model

**Script**: [merge_model.py](merge_model.py)

**Command**:
```bash
uv run .\merge_model.py
```

**Output**: `./merged_cyber_model/` - Full model with LoRA weights merged

### 3. Converting to GGUF Format

**Command**:
```bash
python convert_hf_to_gguf.py ..\..\ml_research\merged_cyber_model --outfile ..\cyber_model.gguf --outtype q8_0
```

**Details**:
- Model architecture: LlamaForCausalLM
- Quantization type: Q8_0 (8-bit quantization)
- Total tensors: 201
- Model size: 1.2 GB
- Context length: 2048
- Embedding dimension: 2048
- Feed-forward dimension: 5632
- Attention heads: 32
- Key-value heads: 4

### 4. Deploying to Ollama

**Modelfile Configuration**: [Modelfile](Modelfile)

**Command**:
```bash
ollama create cyberexpert -f Modelfile
```

**Deployed Model**:
- Name: `cyberexpert:latest`
- ID: b6de9dd627e8
- Size: 1.2 GB

**Usage**:
```bash
ollama list
ollama run cyberexpert
```

## Files in Repository

- **Vis_local_finetune_transformers.py**: Main fine-tuning script with LoRA
- **merge_model.py**: Script to merge LoRA adapters with base model
- **resume_training_dataset.jsonl**: Training dataset in JSONL format
- **Modelfile**: Ollama model configuration
- **finetune_with_transformers_cmd_output.txt**: Complete terminal output log
- **cybersecurity_expert_model_final/**: LoRA adapter weights
- **merged_cyber_model/**: Full merged model
- **pyproject.toml**: Python project dependencies
- **uv.lock**: Locked dependencies

## Evaluation Metrics

**Epoch 1**:
- Eval loss: 1.8062
- Mean token accuracy: 60.09%

**Epoch 2**:
- Eval loss: 1.6054
- Mean token accuracy: 63.66%

**Epoch 3 (Final)**:
- Eval loss: 1.5549
- Mean token accuracy: 64.57%

## Notes

- The training uses gradient checkpointing to reduce memory usage
- Optimized for Windows with `multiprocessing.freeze_support()`
- Uses paged_adamw_8bit optimizer for memory efficiency
- BFloat16 precision leverages Blackwell GPU native capabilities
