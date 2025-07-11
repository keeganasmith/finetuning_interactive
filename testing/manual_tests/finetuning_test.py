import os, sys
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
grandparent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.insert(0, grandparent_dir)
from finetuning import DefaultFineTuner, LoraStrategy
from peft import LoraConfig, TaskType, PeftType
GPT2_PATH = "./example_models/"
def lora_inference_test():
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        peft_type=PeftType.LORA,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        #target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    tuner = DefaultFineTuner(
        model_path="./example_models/gpt2",
        dataset_path="./example_data/train.jsonl",
        dataset_type="json",
        strategy=LoraStrategy(lora_config),
        output_dir="./example_finetuned_models/"
    )

    inference_result = tuner.infer(["What is the sqrt of pi?"])
    print("INFERENCE TEST RESULT")
    print("What is the sqrt of pi?", inference_result)

def lora_training_test():
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        peft_type=PeftType.LORA,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        #target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    tuner = DefaultFineTuner(
        model_path="./example_models/gpt2",
        dataset_path="./example_data/train.jsonl",
        dataset_type="json",
        strategy=LoraStrategy(lora_config),
        output_dir="./example_finetuned_models/"
    )
    tuner.train()

def default_constructor_test():
    #lora_cfg  = LoraConfig(...)            
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        peft_type=PeftType.LORA,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        #target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    tuner = DefaultFineTuner(
        model_path="./example_models/gpt2",
        dataset_path="./example_data/train.jsonl",
        dataset_type="json",
        strategy=LoraStrategy(lora_config),
    )

def main():
    #default_constructor_test()
    #lora_training_test()
    lora_inference_test()

if __name__ == "__main__":
    main()