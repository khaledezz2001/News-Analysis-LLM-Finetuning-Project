# get_data.py
import json
import os
from os.path import join
import random
from tqdm.auto import tqdm
import requests
from datetime import datetime

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from google.colab import drive
from google.colab import userdata
from openai import OpenAI

# Mount Google Drive
drive.mount('/content/drive')

# Install required packages
#!pip install -qU transformers==4.48.3 datasets==3.2.0 openai==1.61.0
#!pip install -qU json-repair==0.29.1
#!pip install json_repair
import json_repair

# Define data structures
StoryCategory = Literal[
    "politics", "sports", "art", "technology", "economy",
    "health", "entertainment", "science", "not_specified"
]

EntityType = Literal[
    "person-male", "person-female", "location", "organization", "event", "time",
    "quantity", "money", "product", "law", "disease", "artifact", "not_specified"
]

class Entity(BaseModel):
    entity_value: str = Field(..., description="The actual name or value of the entity.")
    entity_type: EntityType = Field(..., description="The type of recognized entity.")

class NewsDetails(BaseModel):
    story_title: str = Field(..., min_length=5, max_length=300, description="A fully informative and SEO optimized title of the story.")
    story_keywords: List[str] = Field(..., min_length=1, description="Relevant keywords associated with the story.")
    story_summary: List[str] = Field(..., min_length=1, max_length=5, description="Summarized key points about the story (1-5 points).")
    story_category: StoryCategory = Field(..., description="Category of the news story.")
    story_entities: List[Entity] = Field(..., min_length=1, max_length=10, description="List of identified entities in the story.")

class TranslatedStory(BaseModel):
    translated_title: str = Field(..., min_length=5, max_length=300, description="Suggested translated title of the news story.")
    translated_content: str = Field(..., min_length=5, description="Translated content of the news story.")

def parse_json(text):
    """Helper function to parse JSON with repair"""
    try:
        return json_repair.loads(text)
    except:
        return None

def setup_openai_client():
    """Setup OpenAI client with API keys"""
    return OpenAI(
        api_key=userdata.get('openai-colab'),
        organization=userdata.get('openai-org')
    )

def load_raw_data(data_dir):
    """Load raw news data from JSONL file"""
    raw_data_path = join(data_dir, "datasets", "news-sample.jsonl")
    raw_data = []
    
    for line in open(raw_data_path):
        if line.strip() == "":
            continue
        raw_data.append(json.loads(line.strip()))
    
    random.Random(101).shuffle(raw_data)
    print(f"Loaded {len(raw_data)} raw data samples")
    return raw_data

def create_details_extraction_prompt(story_content):
    """Create prompt for details extraction task"""
    return [
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data parser.",
                "You will be provided by an Arabic text associated with a Pydantic scheme.",
                "Generate the output in the same story language.",
                "You have to extract JSON details from text according the Pydantic details.",
                "Extract details as mentioned in text.",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## Story:",
                story_content.strip(),
                "",
                "## Pydantic Details:",
                json.dumps(NewsDetails.model_json_schema(), ensure_ascii=False),
                "",
                "## Story Details:",
                "```json"
            ])
        }
    ]

def create_translation_prompt(story_content, target_language):
    """Create prompt for translation task"""
    return [
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
                json.dumps(TranslatedStory.model_json_schema(), ensure_ascii=False),
                "",
                "## Targeted Language or Dialect:",
                target_language,
                "",
                "## Story:",
                story_content.strip(),
                "",
                "## Translated Story:",
                "```json"
            ])
        }
    ]

def generate_details_extraction_data(client, raw_data, data_dir):
    """Generate data for details extraction task"""
    cloud_model_id = "gpt-4o-mini"
    price_per_1m_input_tokens = 0.150
    price_per_1m_output_tokens = 0.600
    prompt_tokens = 0
    completion_tokens = 0
    
    save_to = join(data_dir, "datasets", "sft.jsonl")
    
    ix = 0
    for story in tqdm(raw_data, desc="Generating details extraction data"):
        prompt = create_details_extraction_prompt(story['content'])
        
        response = client.chat.completions.create(
            messages=prompt,
            model=cloud_model_id,
            temperature=0.2,
        )
        
        if response.choices[0].finish_reason != "stop":
            prompt_tokens += response.usage.prompt_tokens
            continue
        
        llm_response = response.choices[0].message.content
        llm_resp_dict = parse_json(llm_response)
        
        if not llm_resp_dict:
            continue
        
        with open(save_to, "a", encoding="utf8") as dest:
            dest.write(json.dumps({
                "id": ix,
                "story": story['content'].strip(),
                "task": "Extract the story details into a JSON.",
                "output_scheme": json.dumps(NewsDetails.model_json_schema(), ensure_ascii=False),
                "response": llm_resp_dict,
            }, ensure_ascii=False, default=str) + "\n")
        
        ix += 1
        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens
        
        if (ix % 3) == 0:
            cost_input = (prompt_tokens / 1_000_000) * price_per_1m_input_tokens
            cost_output = (completion_tokens / 1_000_000) * price_per_1m_output_tokens
            total_cost = cost_input + cost_output
            print(f"Iteration {ix}: Total Cost = ${total_cost:.4f}")
    
    return ix

def generate_translation_data(client, raw_data, data_dir):
    """Generate data for translation task"""
    cloud_model_id = "gpt-4o-mini"
    price_per_1m_input_tokens = 0.150
    price_per_1m_output_tokens = 0.600
    prompt_tokens = 0
    completion_tokens = 0
    
    save_to = join(data_dir, "datasets", "sft.jsonl")
    
    ix = 0
    for story in tqdm(raw_data, desc="Generating translation data"):
        for targeted_lang in ["English", "French"]:
            prompt = create_translation_prompt(story['content'], targeted_lang)
            
            response = client.chat.completions.create(
                messages=prompt,
                model=cloud_model_id,
                temperature=0.2,
            )
            
            if response.choices[0].finish_reason != "stop":
                prompt_tokens += response.usage.prompt_tokens
                continue
            
            llm_response = response.choices[0].message.content
            llm_resp_dict = parse_json(llm_response)
            
            if not llm_resp_dict:
                continue
            
            with open(save_to, "a", encoding="utf8") as dest:
                dest.write(json.dumps({
                    "id": ix,
                    "story": story['content'].strip(),
                    "output_scheme": json.dumps(TranslatedStory.model_json_schema(), ensure_ascii=False),
                    "task": f"You have to translate the story content into {targeted_lang} associated with a title into a JSON.",
                    "response": llm_resp_dict,
                }, ensure_ascii=False, default=str) + "\n")
            
            ix += 1
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            
            if (ix % 3) == 0:
                cost_input = (prompt_tokens / 1_000_000) * price_per_1m_input_tokens
                cost_output = (completion_tokens / 1_000_000) * price_per_1m_output_tokens
                total_cost = cost_input + cost_output
                print(f"Iteration {ix}: Total Cost = ${total_cost:.4f}")
    
    return ix

def prepare_finetuning_data(data_dir):
    """Prepare data in LLaMA-Factory format"""
    sft_data_path = join(data_dir, "datasets", "sft.jsonl")
    llm_finetuning_data = []
    
    system_message = "\n".join([
        "You are a professional NLP data parser.",
        "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
        "Do not generate any introduction or conclusion."
    ])
    
    for line in open(sft_data_path):
        if line.strip() == "":
            continue
        
        rec = json.loads(line.strip())
        
        llm_finetuning_data.append({
            "system": system_message,
            "instruction": "\n".join([
                "# Story:",
                rec["story"],
                "# Task:",
                rec["task"],
                "# Output Scheme:",
                rec["output_scheme"],
                "",
                "# Output JSON:",
                "```json"
            ]),
            "input": "",
            "output": "\n".join([
                "```json",
                json.dumps(rec["response"], ensure_ascii=False, default=str),
                "```"
            ]),
            "history": []
        })
    
    random.Random(101).shuffle(llm_finetuning_data)
    return llm_finetuning_data

def save_datasets(data_dir, llm_finetuning_data, train_sample_sz=2700):
    """Save training and validation datasets"""
    train_ds = llm_finetuning_data[:train_sample_sz]
    eval_ds = llm_finetuning_data[train_sample_sz:]
    
    os.makedirs(join(data_dir, "datasets", "llamafactory-finetune-data"), exist_ok=True)
    
    with open(join(data_dir, "datasets", "llamafactory-finetune-data", "train.json"), "w", encoding="utf8") as dest:
        json.dump(train_ds, dest, ensure_ascii=False, default=str)
    
    with open(join(data_dir, "datasets", "llamafactory-finetune-data", "val.json"), "w", encoding="utf8") as dest:
        json.dump(eval_ds, dest, ensure_ascii=False, default=str)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(eval_ds)}")

def main():
    """Main function to run data preparation pipeline"""
    data_dir = "/gdrive/MyDrive/LLM/llm-finetuning"
    
    # Setup
    client = setup_openai_client()
    raw_data = load_raw_data(data_dir)
    
    # Generate training data
    print("Starting data generation...")
    details_count = generate_details_extraction_data(client, raw_data[:100], data_dir)  # Limit to 100 for demo
    translation_count = generate_translation_data(client, raw_data[:100], data_dir)  # Limit to 100 for demo
    
    print(f"Generated {details_count} details extraction samples")
    print(f"Generated {translation_count} translation samples")
    
    # Prepare finetuning data
    finetuning_data = prepare_finetuning_data(data_dir)
    save_datasets(data_dir, finetuning_data)
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()