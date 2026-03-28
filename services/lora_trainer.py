"""
services/lora_trainer.py

Trains a LoRA adapter on YOUR Drupal nodes from pgvector DB.
Run this once as a script — not part of the API.

Usage:
    docker exec -it vector-api python scripts/train_lora.py

CPU friendly — uses TinyLlama 1.1B which works without GPU.
Training time on CPU: ~10-30 minutes depending on node count.
"""

import os
import json
import psycopg2
import psycopg2.extras
from config import DATABASE_URL

# ── LoRA output folder ────────────────────────────────────────
LORA_ADAPTER_PATH = "/app/lora_adapter"
BASE_MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def fetch_drupal_nodes() -> list:
    """
    Fetch all stored nodes from pgvector DB.
    These become your LoRA training data.
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT node_id, title, content
                FROM node_vectors
                WHERE content IS NOT NULL
                AND LENGTH(content) > 50
                ORDER BY node_id;
            """)
            rows = cur.fetchall()

    print(f"Found {len(rows)} nodes for training")
    return [dict(row) for row in rows]


def build_training_data(nodes: list) -> list:
    """
    Convert Drupal nodes into LoRA training format.

    Each node becomes multiple training examples:
    1. Direct question about title
    2. Content question
    3. What is this about question

    More examples = better training.
    """
    training_data = []

    for node in nodes:
        title   = node["title"]
        content = node["content"]

        # Example 1 — direct title question
        training_data.append({
            "instruction": f"Tell me about: {title}",
            "input":       "",
            "output":      content,
        })

        # Example 2 — what is question
        training_data.append({
            "instruction": f"What is {title}?",
            "input":       "",
            "output":      content,
        })

        # Example 3 — explain question
        training_data.append({
            "instruction": f"Explain {title} in detail.",
            "input":       "",
            "output":      content,
        })

    print(f"Built {len(training_data)} training examples from {len(nodes)} nodes")
    return training_data


def format_prompt(example: dict) -> str:
    """
    Format training example into Alpaca prompt format.
    Model learns: instruction → output mapping.
    """
    return f"""Below is an instruction. Write a helpful response.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""


def train_lora(nodes: list):
    """
    Main LoRA training function.
    Trains TinyLlama on your Drupal nodes.
    """
    # Import here — heavy imports only when training
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import DataLoader

    print("Loading TinyLlama tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading TinyLlama model (CPU mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map = "cpu",    # CPU only — no GPU needed
        low_cpu_mem_usage = True,
    )

    # ── Add LoRA adapters ─────────────────────────────────────
    print("Adding LoRA adapters...")
    lora_config = LoraConfig(
        r              = 8,          # rank — lower = faster on CPU
        lora_alpha     = 16,
        target_modules = ["q_proj", "v_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Build dataset ─────────────────────────────────────────
    print("Building training dataset...")
    training_data = build_training_data(nodes)

    # Format prompts
    formatted = [
        {"text": format_prompt(example)}
        for example in training_data
    ]

    dataset = Dataset.from_list(formatted)

    # Tokenize
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation  = True,
            max_length  = 512,
            padding     = "max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched = True,
        remove_columns = ["text"],
    )
    tokenized_dataset = tokenized_dataset.with_format("torch")

    # ── Training arguments — optimized for CPU ────────────────
    training_args = TrainingArguments(
        output_dir                  = LORA_ADAPTER_PATH,
        num_train_epochs            = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1,   # ← reduce from 2 to 1
        learning_rate               = 2e-4,
        logging_steps               = 2,
        save_steps                  = 10,
        max_steps                   = 10,  # ← reduce from 50 to 10
        fp16                        = False,
        use_cpu                     = True,
        dataloader_num_workers      = 0,
        report_to                   = "none",
    )

    from transformers import Trainer

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = tokenized_dataset,
        data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm = False
        ),
    )

    print("Starting LoRA training on CPU...")
    print("This may take 10-30 minutes depending on node count...")
    trainer.train()

    # ── Save only LoRA adapter (small file ~16MB) ─────────────
    print(f"Saving LoRA adapter to {LORA_ADAPTER_PATH}...")
    model.save_pretrained(LORA_ADAPTER_PATH)
    tokenizer.save_pretrained(LORA_ADAPTER_PATH)

    # Save metadata
    metadata = {
        "base_model":    BASE_MODEL_NAME,
        "nodes_trained": len(nodes),
        "examples":      len(training_data),
        "adapter_path":  LORA_ADAPTER_PATH,
    }
    with open(f"{LORA_ADAPTER_PATH}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("LoRA training complete!")
    print(f"Adapter saved to: {LORA_ADAPTER_PATH}")
    return LORA_ADAPTER_PATH


def run_training():
    """Entry point — fetch nodes and train."""
    nodes = fetch_drupal_nodes()

    if len(nodes) == 0:
        print("No nodes found in pgvector DB!")
        print("Store some nodes first using POST /nodes/store")
        return

    if len(nodes) < 5:
        print(f"Warning: Only {len(nodes)} nodes found.")
        print("LoRA works better with 50+ nodes.")
        print("Continuing anyway...")

    train_lora(nodes)