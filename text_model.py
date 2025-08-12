# src/text_model.py
import os
from transformers import pipeline

# default model dir (optional)
DEFAULT_MODEL_DIR = os.path.join("models", "text_model")

def load_text_model(path: str = DEFAULT_MODEL_DIR):
    """
    Returns a Hugging Face pipeline object for text classification.
    If a local folder exists at `path`, it will be used; otherwise it'll download the default model.
    """
    if os.path.exists(path) and os.path.isdir(path):
        clf = pipeline("text-classification", model=path, return_all_scores=True, framework="pt")
        return clf
    else:
        # download a lightweight emotion model
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        clf = pipeline("text-classification", model=model_name, return_all_scores=True, framework="pt")
        # optionally save for offline use
        try:
            os.makedirs(path, exist_ok=True)
            clf.model.save_pretrained(path)
            clf.tokenizer.save_pretrained(path)
        except Exception:
            pass
        return clf

def predict_text_mood(model, text: str) -> str:
    """
    model: pipeline returned by load_text_model
    returns predicted label (string) — canonicalized lower-case
    """
    if text is None or text.strip() == "":
        return "neutral"
    preds = model(text)
    # preds is a list of lists when return_all_scores=True, or list of dicts when False
    if isinstance(preds, list) and len(preds) > 0:
        # first item is list of dicts
        scores = preds[0]
        top = max(scores, key=lambda x: x['score'])
        return top['label'].lower()
    else:
        return "neutral"
