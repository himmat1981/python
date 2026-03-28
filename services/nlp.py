from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import threading

# ── Thread locks — prevent loading model twice if two requests come at same time
_sum_lock = threading.Lock()
_mod_lock = threading.Lock()

# ── Model instances ────────────────────────────────────────────
_tokenizer  = None
_sum_model  = None
_moderator  = None


def get_summarizer():
    """
    Lazy load summarization model.
    Loaded only when first request comes in — not at startup.
    This prevents uvicorn restart loop caused by heavy model loading.
    """
    global _tokenizer, _sum_model
    if _tokenizer is None:
        with _sum_lock:
            if _tokenizer is None:  # double check inside lock
                print("Loading summarization model...")
                _tokenizer = AutoTokenizer.from_pretrained(
                    "Falconsai/text_summarization"
                )
                _sum_model = AutoModelForSeq2SeqLM.from_pretrained(
                    "Falconsai/text_summarization"
                )
                print("Summarization model loaded!")
    return _tokenizer, _sum_model


def get_moderator():
    """
    Lazy load moderation model.
    Loaded only when first request comes in — not at startup.
    """
    global _moderator
    if _moderator is None:
        with _mod_lock:
            if _moderator is None:  # double check inside lock
                print("Loading moderation model...")
                _moderator = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert"
                )
                print("Moderation model loaded!")
    return _moderator


def summarize_text(text: str) -> str:
    try:
        tokenizer, model = get_summarizer()

        inputs = tokenizer(
            "summarize: " + text,
            return_tensors = "pt",
            max_length     = 512,
            truncation     = True,
        )

        outputs = model.generate(
            inputs["input_ids"],
            max_length     = 120,
            min_length     = 30,
            length_penalty = 2.0,
            num_beams      = 4,
            early_stopping = True,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        return f"Error: {str(e)}"


def moderate_text(text: str) -> dict:
    try:
        result   = get_moderator()(text)[0]
        label    = result["label"]
        score    = float(result["score"])
        is_toxic = label.lower() == "toxic" and score > 0.7

        return {
            "label":    label,
            "score":    score,
            "is_toxic": is_toxic,
            "verdict":  "BLOCKED" if is_toxic else "CLEAN"
        }
    except Exception as e:
        return {
            "label":    "error",
            "score":    0.0,
            "is_toxic": False,
            "verdict":  "ERROR",
            "error":    str(e)
        }