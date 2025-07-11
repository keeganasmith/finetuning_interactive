import os
import torch
from datasets import load_dataset
from huggingface_hub import list_datasets
print(list_datasets())
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup



class FineTune:
    def __init__(self, model_path: str, dataset_path, dataset_type: str):
        self.accelerator = Accelerator()

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Enable LoRA (parameter-efficient fine-tuning)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # may vary depending on model
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, peft_config)

        # Load dataset
        raw_dataset = load_dataset(dataset_type, data_files=dataset_path)

        def tokenize_function(example):
            return self.tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
        self.dataloader = DataLoader(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=4,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )
        )

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=1000  # adjust based on dataset size
        )

        # Prepare all with accelerator
        self.model, self.dataloader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.dataloader, self.optimizer, self.lr_scheduler
        )

        


