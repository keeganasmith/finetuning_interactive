from flask import Flask, request, jsonify
from finetuning import LoraStrategy, DefaultFineTuner
from peft import TaskType, PeftType

app = Flask(__name__)
TASK_TYPE_MAPPING = {
    "CAUSAL_LM": TaskType.CAUSAL_LM
}
PEFT_TYPE_MAPPING = {
    "LORA": PeftType.LORA
}
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    Expects JSON payload:
      {
        "model_path": "..."," +
        "dataset_path": "...",
        "dataset_type": "json",
        "lora_config": { ... }, https://huggingface.co/docs/peft/en/package_reference/lora
        "task_type": Literal["CAUSAL_LM"],
        "peft_type": Literal["LORA"],
        "output_dir": "...",
        "batch_size": 4,
        "max_length": 512,
        "lr": 5e-5,
        "warmup_steps": 100,
        "total_steps": 1000
      }
    """
    payload = request.get_json()
    try:
        # Unpack payload
        model_path   = payload['model_path']
        dataset_path = payload['dataset_path']
        dataset_type = payload['dataset_type']
        lora_cfg     = payload['lora_config']
        output_dir   = payload['output_dir']
        batch_size   = payload.get('batch_size', 4)
        max_length   = payload.get('max_length', 512)
        lr           = payload.get('lr', 5e-5)
        warmup_steps = payload.get('warmup_steps', 100)
        total_steps  = payload.get('total_steps', 1000)
        task_type = payload["task_type"]
        peft_type = payload["peft_type"]
        lora_cfg["task_type"] = TASK_TYPE_MAPPING[task_type]
        lora_cfg["peft_type"] = TASK_TYPE_MAPPING[peft_type]
        # Build and run tuner
        strategy = LoraStrategy(lora_cfg)
        tuner    = DefaultFineTuner(
            model_path=model_path,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            strategy=strategy,
            output_dir=output_dir,
            batch_size=batch_size,
            max_length=max_length,
            lr=lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        tuner.train()

        return jsonify({
            'status': 'success',
            'message': f'Training completed. Model saved to {output_dir}'
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

    