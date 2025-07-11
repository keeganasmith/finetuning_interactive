import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name: str, output_dir: str):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print(f"  → Tokenizer saved to {output_dir}/tokenizer\n")

    print(f"Downloading model '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print(f"  → Model saved to {output_dir}/model\n")

    print("Download complete!")

def main():
    model_name = "gpt2"
    output_dir = "./example_models/gpt2"

    download_model(model_name, output_dir)

if __name__ == "__main__":
    main()
