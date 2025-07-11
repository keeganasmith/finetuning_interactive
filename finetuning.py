from abc import ABC, abstractmethod
from typing import Any, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import get_peft_model
# --- Strategy Pattern for PEFT ---
class FinetuningStrategy(ABC):
    @abstractmethod
    def apply(self, model: Any) -> Any:
        """Apply the chosen PEFT method to the base model."""
        ...

class LoraStrategy(FinetuningStrategy):
    def __init__(self, lora_config):
        self.lora_config = lora_config

    def apply(self, model):
        # get_peft_model imported from peft
        return get_peft_model(model, self.lora_config)

# --- Abstract Base FineTuner ---
class BaseFineTuner(ABC):
    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        dataset_type: str,
        strategy: FinetuningStrategy,
        batch_size: int = 4,
        max_length: int = 512,
        lr: float = 5e-5,
        warmup_steps: int = 100,
        total_steps: int = 1000,
    ):
        self.accelerator    = Accelerator()
        self.model_path     = model_path
        self.dataset_path   = dataset_path
        self.dataset_type   = dataset_type
        self.strategy       = strategy
        self.batch_size     = batch_size
        self.max_length     = max_length
        self.lr             = lr
        self.warmup_steps   = warmup_steps
        self.total_steps    = total_steps

        self.tokenizer      = self.build_tokenizer()
        self.model          = self.build_model()
        self.dataset        = self.build_dataset()
        self.dataloader     = self.build_dataloader()
        self.optimizer, self.lr_scheduler = self.build_optim_and_scheduler()

        # wrap with accelerator
        self.model, self.dataloader, self.optimizer, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model,
                self.dataloader,
                self.optimizer,
                self.lr_scheduler,
            )

    @abstractmethod
    def build_tokenizer(self) -> Any:
        """Load tokenizer from self.model_path."""
        ...

    def build_model(self) -> Any:
        """Load base model and apply finetuning strategy."""
        base = AutoModelForCausalLM.from_pretrained(self.model_path)
        return self.strategy.apply(base)

    def build_dataset(self) -> Any:
        """Load and tokenize dataset."""
        raw = load_dataset(self.dataset_type, data_files=self.dataset_path)
        def tok_fn(ex):
            return self.tokenizer(
                ex["text"], truncation=True,
                padding="max_length", max_length=self.max_length
            )
        tok = raw.map(tok_fn, batched=True)
        return tok["train"]

    def build_dataloader(self) -> DataLoader:
        """Wrap dataset in DataLoader with MLM collator disabled."""
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        return DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=collator,
        )

    def build_optim_and_scheduler(self) -> Tuple[Optimizer, Any]:
        """Configure optimizer and LR scheduler."""
        opt = AdamW(self.model.parameters(), lr=self.lr)
        sched = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        return opt, sched


class DefaultFineTuner(BaseFineTuner):
    def build_tokenizer(self) -> Any:
        # Load a pretrained tokenizer from model_path
        result = AutoTokenizer.from_pretrained(self.model_path)
        result.pad_token = result.eos_token
        return result

    # build_model is inherited — it loads the base LM and applies your strategy

    # build_dataset is inherited — it loads & tokenizes via self.tokenizer

    # build_dataloader is inherited — wraps the dataset in a DataLoader

    def build_optim_and_scheduler(self) -> Tuple[Optimizer, Any]:
        # If you want different defaults, override here; otherwise
        # you can just call super():
        return super().build_optim_and_scheduler()
