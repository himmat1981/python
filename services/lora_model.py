"""
services/lora_model.py

Loads trained LoRA adapter and answers questions.
Used by chatbot router as first attempt before RAG.

Singleton pattern — model loaded once, reused for all requests.
"""

import os
import torch
import torch.nn.functional as F
from typing import Optional

LORA_ADAPTER_PATH = "/app/lora_adapter"
BASE_MODEL_NAME   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Singletons ────────────────────────────────────────────────
_model     = None
_tokenizer = None


def is_lora_trained() -> bool:
    """Check if LoRA adapter exists — was training done?"""
    return os.path.exists(LORA_ADAPTER_PATH) and \
           os.path.exists(f"{LORA_ADAPTER_PATH}/adapter_config.json")


def get_lora_model():
    """
    Load LoRA model once — singleton pattern.
    Returns None if not trained yet.
    """
    global _model, _tokenizer

    if not is_lora_trained():
        return None, None

    if _model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print("Loading TinyLlama base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map        = "cpu",
            low_cpu_mem_usage = True,
        )

        print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
        _model = PeftModel.from_pretrained(
            base_model,
            LORA_ADAPTER_PATH,
        )
        _model.eval()

        _tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)
        _tokenizer.pad_token = _tokenizer.eos_token

        print("LoRA model ready!")

    return _model, _tokenizer


def lora_answer(question: str) -> dict:
    """
    Get answer from LoRA trained model WITH confidence score.

    Returns:
        answer:     the generated text
        confidence: 0.0 to 1.0 — how sure model is
        available:  False if LoRA not trained yet
    """
    model, tokenizer = get_lora_model()

    # ── LoRA not trained yet ──────────────────────────────────
    if model is None:
        return {
            "answer":     "",
            "confidence": 0.0,
            "available":  False,
        }

    # ── Format prompt — same format as training ───────────────
    prompt = f"""Below is an instruction. Write a helpful response.

### Instruction:
{question}

### Input:


### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 512,
    )

    # ── Generate with probability scores ─────────────────────
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens          = 200,
            temperature             = 0.7,
            do_sample               = True,
            output_scores           = True,    # get token probabilities
            return_dict_in_generate = True,
            pad_token_id            = tokenizer.eos_token_id,
        )

    # ── Decode answer ─────────────────────────────────────────
    full_text = tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens = True,
    )

    # Extract only the response part
    if "### Response:" in full_text:
        answer = full_text.split("### Response:")[-1].strip()
    else:
        answer = full_text.strip()

    # ── Calculate confidence score ────────────────────────────
    # Average probability of each generated token
    # High = model was confident about each word
    # Low  = model was guessing
    confidence = 0.5  # default
    if outputs.scores:
        token_probs = []
        for score in outputs.scores:
            probs    = F.softmax(score, dim=-1)
            max_prob = probs.max().item()
            token_probs.append(max_prob)
        confidence = sum(token_probs) / len(token_probs)

    return {
        "answer":     answer,
        "confidence": round(confidence, 3),
        "available":  True,
    }


def check_lora_confidence(lora_result: dict) -> bool:
    """
    Check if LoRA answer is good enough to use.
    Returns True = use LoRA answer
    Returns False = fallback to RAG

    Three checks:
    1. Confidence score threshold
    2. Uncertainty phrases in answer
    3. Answer too short
    """
    if not lora_result["available"]:
        return False

    answer     = lora_result["answer"]
    confidence = lora_result["confidence"]

    # Check 1 — confidence threshold
    CONFIDENCE_THRESHOLD = 0.75
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"LoRA low confidence: {confidence} < {CONFIDENCE_THRESHOLD}")
        return False

    # Check 2 — uncertainty phrases
    uncertainty_phrases = [
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i cannot find",
        "no information",
        "not available",
        "i don't have",
        "cannot answer",
        "i am unable",
        "no relevant",
    ]
    answer_lower = answer.lower()
    for phrase in uncertainty_phrases:
        if phrase in answer_lower:
            print(f"LoRA uncertainty phrase found: '{phrase}'")
            return False

    # Check 3 — answer too short
    if len(answer.split()) < 10:
        print(f"LoRA answer too short: {len(answer.split())} words")
        return False

    print(f"LoRA answer accepted — confidence: {confidence}")
    return True