# finetune_llm.py
import os
import json
from os.path import join
import subprocess

def setup_environment():
    """Setup environment and install required packages"""
    print("Setting up environment...")
    
    # Install LLaMA-Factory
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hiyouga/LLaMA-Factory.git"])
    subprocess.run(["cd", "LLaMA-Factory", "&&", "pip", "install", "-e", "."], shell=True)
    
    # Install additional packages
    subprocess.run(["pip", "install", "-qU", "wandb"])
    subprocess.run(["pip", "install", "-qU", "vllm==0.7.2"])
    
    # Login to wandb and huggingface (requires user data)
    import wandb
    from google.colab import userdata
    
    wandb.login(key=userdata.get('wandb'))
    hf_token = userdata.get('hfToken')
    subprocess.run(f"huggingface-cli login --token {hf_token}", shell=True)

def create_config_file(data_dir):
    """Create LLaMA-Factory configuration file"""
    config_content = f"""
### model
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_target: all

### dataset
dataset: news_finetune_train
eval_dataset: news_finetune_val
template: qwen
cutoff_len: 3500
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {data_dir}/models/
logging_steps: 10
save_steps: 500
plot_loss: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100

report_to: wandb
run_name: newsx-finetune-llamafactory

push_to_hub: true
export_hub_model_id: "bakrianoo/news-analyzer"
hub_private_repo: true
hub_strategy: checkpoint
"""
    
    config_path = "/content/LLaMA-Factory/examples/train_lora/news_finetune.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Configuration file created at: {config_path}")

def create_dataset_info(data_dir):
    """Create dataset info for LLaMA-Factory"""
    dataset_info = {
        "news_finetune_train": {
            "file_name": f"{data_dir}/datasets/llamafactory-finetune-data/train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history"
            }
        },
        "news_finetune_val": {
            "file_name": f"{data_dir}/datasets/llamafactory-finetune-data/val.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "history": "history"
            }
        }
    }
    
    # Read existing dataset info
    dataset_info_path = "/content/LLaMA-Factory/data/dataset_info.json"
    try:
        with open(dataset_info_path, "r") as f:
            existing_info = json.load(f)
    except FileNotFoundError:
        existing_info = {}
    
    # Update with new datasets
    existing_info.update(dataset_info)
    
    with open(dataset_info_path, "w") as f:
        json.dump(existing_info, f, indent=2)
    
    print(f"Dataset info updated at: {dataset_info_path}")

def run_training():
    """Run the training process"""
    print("Starting training...")
    subprocess.run([
        "cd", "LLaMA-Factory/", "&&", 
        "llamafactory-cli", "train", 
        "/content/LLaMA-Factory/examples/train_lora/news_finetune.yaml"
    ], shell=True)

def setup_inference(data_dir):
    """Setup the model for inference with vLLM"""
    print("Setting up inference server...")
    
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_model_id = f"{data_dir}/models"
    
    # Start vLLM server with LoRA adapter
    cmd = f"""
    nohup vllm serve "{base_model_id}" \
    --dtype=half \
    --gpu-memory-utilization 0.8 \
    --max_lora_rank 64 \
    --enable-lora \
    --lora-modules news-lora="{adapter_model_id}" &
    """
    
    subprocess.run(cmd, shell=True)
    print("vLLM server started in background")
    
    # Wait for server to start
    import time
    time.sleep(10)
    
    # Check server status
    result = subprocess.run(["tail", "-n", "30", "nohup.out"], capture_output=True, text=True)
    print("Server output:")
    print(result.stdout)

def test_inference():
    """Test the finetuned model with a sample query"""
    print("Testing inference...")
    
    from transformers import AutoTokenizer
    import requests
    
    # Sample translation messages
    story = "ذكرت مجلة فوربس أن العائلة تلعب دورا محوريا في تشكيل علاقة الأفراد بالمال..."
    
    translation_messages = [
        {
            "role": "system",
            "content": "\n".join([
                "You are a professional translator.",
                "You will be provided by an Arabic text.",
                "You have to translate the text into the `Targeted Language`.",
                "Follow the provided Scheme to generate a JSON",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## Pydantic Details:",
                json.dumps({
                    "title": "translated_title",
                    "content": "translated_content"
                }),
                "",
                "## Targeted Language or Dialect:",
                "English",
                "",
                "## Story:",
                story,
                "",
                "## Translated Story:",
                "```json"
            ])
        }
    ]
    
    # Apply chat template
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    prompt = tokenizer.apply_chat_template(
        translation_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Send request to vLLM server
    response = requests.post("http://localhost:8000/v1/completions", json={
        "model": "news-lora",
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.3
    })
    
    if response.status_code == 200:
        result = response.json()
        print("Inference test successful!")
        print("Response:", result.get("choices", [{}])[0].get("text", ""))
    else:
        print(f"Inference test failed with status code: {response.status_code}")

def main():
    """Main function to run the finetuning pipeline"""
    data_dir = "/gdrive/MyDrive/LLM/llm-finetuning"
    
    # Setup environment
    setup_environment()
    
    # Create configuration
    create_config_file(data_dir)
    create_dataset_info(data_dir)
    
    # Run training
    run_training()
    
    # Setup and test inference
    setup_inference(data_dir)
    test_inference()
    
    print("Finetuning pipeline completed!")

if __name__ == "__main__":
    main()