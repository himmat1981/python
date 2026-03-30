"""
scripts/train_lora.py

Run this ONCE to train LoRA on your Drupal nodes.

Usage inside Docker container:
    docker exec -it vector-api python scripts/train_lora.py

What it does:
1. Fetches all nodes from pgvector DB
2. Converts them to training examples
3. Trains TinyLlama with LoRA adapters
4. Saves adapter to /app/lora_adapter/

After training:
- Chatbot will try LoRA first
- Falls back to RAG if LoRA not confident
"""

import sys
import os

# Add /app to path so imports work
sys.path.insert(0, "/app")

from services.lora_trainer import run_training

if __name__ == "__main__":
    print("=" * 50)
    print("LoRA Training Script")
    print("Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print("Device: CPU")
    print("=" * 50)
    run_training()
    print("=" * 50)
    print("Training complete!")
    print("Restart python-api to load LoRA model:")
    print("docker-compose -f docker-compose.dev.yml restart python-api")
    print("=" * 50)