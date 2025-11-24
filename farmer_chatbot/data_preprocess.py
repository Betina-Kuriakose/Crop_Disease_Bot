"""
Utility script to clean the questions dataset:
- Lowercasing and punctuation removal
- Tokenization
- Stop-word removal
- Lemmatization

Outputs a CSV with the transformed text so the NLP pipeline can reuse it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import nltk
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

DATA_PATH = Path("archive/questionsv4.csv")
OUTPUT_PATH = Path("archive/questions_cleaned.csv")
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}


def ensure_nltk_assets() -> None:
    """Download required NLTK resources if missing."""
    for pkg, locator in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(locator)
        except (LookupError, OSError):
            nltk.download(pkg)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str], stop_words: set[str]) -> list[str]:
    return [tok for tok in tokens if tok not in stop_words]


def lemmatize(tokens: list[str], lemmatizer: WordNetLemmatizer) -> list[str]:
    return [lemmatizer.lemmatize(tok) for tok in tokens]


def main() -> None:
    ensure_nltk_assets()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["questions"] = df["questions"].fillna("")

    df["question_clean"] = df["questions"].apply(normalize_text)
    df["tokens"] = df["question_clean"].apply(tokenize)
    df["tokens_no_stop"] = df["tokens"].apply(lambda toks: remove_stopwords(toks, stop_words))
    df["tokens_lemmatized"] = df["tokens_no_stop"].apply(lambda toks: lemmatize(toks, lemmatizer))
    df["question_processed"] = df["tokens_lemmatized"].apply(lambda toks: " ".join(toks))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote cleaned dataset with {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


