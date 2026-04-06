from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import threading

# ── Thread locks — prevent loading model twice if two requests come at same time
_sum_lock = threading.Lock()
_mod_lock = threading.Lock()

# ── Chain / model instances ────────────────────────────────────
_sum_chain = None
_moderator = None


def get_summarizer_chain():
    """
    Lazy-load the summarization model and build a LangChain LCEL chain:
      PromptTemplate | HuggingFacePipeline | StrOutputParser
    Loaded only on first request to avoid blocking uvicorn startup.
    """
    global _sum_chain
    if _sum_chain is None:
        with _sum_lock:
            if _sum_chain is None:
                print("Loading summarization model...")
                tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
                model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

                hf_pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=120,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                )

                llm = HuggingFacePipeline(pipeline=hf_pipe)
                prompt = PromptTemplate.from_template("summarize: {text}")
                _sum_chain = prompt | llm | StrOutputParser()
                print("Summarization chain ready!")
    return _sum_chain


def get_moderator():
    """
    Lazy-load moderation model.
    Loaded only when first request comes in — not at startup.
    """
    global _moderator
    if _moderator is None:
        with _mod_lock:
            if _moderator is None:
                print("Loading moderation model...")
                _moderator = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert"
                )
                print("Moderation model loaded!")
    return _moderator


def summarize_text(text: str) -> str:
    try:
        chain = get_summarizer_chain()
        return chain.invoke({"text": text})
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
