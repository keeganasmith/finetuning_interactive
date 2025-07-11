#!/bin/bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./example_models/gpt2",
    "dataset_path": "./example_data/train.jsonl",
    "dataset_type": "json",
    "lora_config": {
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.1,
      "target_modules": ["q_proj","v_proj"]
    },
    "task_type": "CAUSAL_LM",
    "peft_type": "LORA",
    "output_dir": "./example_finetuned_models/opt-lora",
    "batch_size": 4,
    "max_length": 512,
    "lr": 5e-5,
    "warmup_steps": 100,
    "total_steps": 1000
  }'
