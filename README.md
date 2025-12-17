# LLM Fine-Tuning for Arabic News Processing

This project fine-tunes the **Qwen2.5-1.5B** language model to process Arabic news articles. The model is trained to:

* Extract structured information from Arabic news
* Translate Arabic news into English and French

The training uses **LoRA fine-tuning**, and inference is served efficiently using **vLLM**.

---

## Features

* 📰 **Arabic News Understanding**: Extracts title, keywords, category, and named entities
* 🌍 **Multilingual Translation**: Translates Arabic news into English and French
* ⚡ **Efficient Training**: Lightweight LoRA fine-tuning
* 🚀 **Fast Inference**: Served using vLLM

---

## Quick Setup

### 1. Install Required Packages

```bash
pip install transformers datasets torch accelerate openai pydantic json-repair
```

### 2. Clone and Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
cd ..
```

### 3. Install vLLM (for Fast Inference)

```bash
pip install vllm
```

---

## Quick Start

### Step 1: Prepare the Dataset

```bash
python get_data.py
```

This script:

* Loads Arabic news articles from `datasets/news-sample.jsonl`
* Uses **GPT-4o** to generate structured training examples
* Saves the final supervised fine-tuning dataset to `datasets/sft.jsonl`

---

### Step 2: Fine-Tune the Model

```bash
python finetune_llm.py
```

This step:

* Fine-tunes **Qwen2.5-1.5B** using **LoRA**
* Saves the trained model to the `models/` directory
* Starts a **vLLM inference server** at `http://localhost:8000`

---

## What the Model Does

### 1. Extract News Details

**Input:**

```text
Arabic news article
```

**Output:**

```json
{
  "title": "...",
  "keywords": ["..."],
  "category": "...",
  "entities": ["..."]
}
```

---

### 2. Translate News

**Input:**

```text
Arabic text
```

**Output:**

```json
{
  "english": "...",
  "french": "..."
}
```

---

## Output Directories

* `datasets/` – Raw and processed datasets
* `models/` – Fine-tuned model checkpoints

---

## Notes

* Ensure you have access to **GPT-4o** for dataset generation
* A GPU is recommended for fine-tuning and vLLM inference

---

## License

This project is intended for research and educational purposes.
