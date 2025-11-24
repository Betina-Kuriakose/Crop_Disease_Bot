"""
Utility helpers for advanced NLP pre-processing, summarization, and
multilingual translation used by the Flask chatbot.

The logic mirrors the experimentation done inside `farmbotmultilingual.ipynb`
without depending on that notebook at runtime.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import nltk
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

try:
    from transformers import (  # type: ignore
        MBart50TokenizerFast,
        MBartForConditionalGeneration,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import guard
    TRANSFORMERS_AVAILABLE = False
    pipeline = None  # type: ignore
    MBartForConditionalGeneration = None  # type: ignore
    MBart50TokenizerFast = None  # type: ignore
    print(f"Transformers not available: {exc}")


SUPPORTED_MBART_LANGS = {
    "en": "en_XX",
    "hi": "hi_IN",
}


def _ensure_nltk_assets() -> None:
    """Download required NLTK corpora once."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",  # Newer nltk versions
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except (LookupError, OSError):  # pragma: no cover - runtime download
            nltk.download(pkg)


class NLPProcessor:
    """Encapsulates summarization, cleaning, and translation helpers."""

    def __init__(
        self,
        summary_model: str = "sshleifer/distilbart-cnn-12-6",
        translator_model: str = "facebook/mbart-large-50-many-to-many-mmt",
        summary_threshold: int = 25,
    ) -> None:
        _ensure_nltk_assets()
        self.summary_threshold = summary_threshold
        self.summary_model = summary_model
        self.translator_model = translator_model
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.summarizer = self._load_summarizer()
        self.translation_model, self.translation_tokenizer = self._load_translator()

    def _load_summarizer(self):
        if not TRANSFORMERS_AVAILABLE:
            return None
        try:
            return pipeline(
                "summarization",
                model=self.summary_model,
                tokenizer=self.summary_model,
                truncation=True,
            )
        except Exception as exc:  # pragma: no cover - hardware/network guard
            print(f"Summarizer unavailable: {exc}")
            return None

    def _load_translator(self):
        if not TRANSFORMERS_AVAILABLE:
            return None, None
        try:
            model = MBartForConditionalGeneration.from_pretrained(
                self.translator_model
            )
            tokenizer = MBart50TokenizerFast.from_pretrained(self.translator_model)
            return model, tokenizer
        except Exception as exc:  # pragma: no cover - hardware/network guard
            print(f"Translation model unavailable: {exc}")
            return None, None

    def normalize_question(self, text: str) -> str:
        """Summarize long queries and clean the text for vectorizer use."""
        if not text:
            return ""
        summarized = self._summarize_if_needed(text.strip())
        return self.clean_text(summarized)

    def clean_text(self, text: str) -> str:
        """Lowercase, remove punctuation/stopwords, and lemmatize."""
        if not text:
            return ""
        # Remove punctuation similar to notebook helper
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in word_tokenize(text)
            if tok not in self.stop_words
        ]
        cleaned = re.sub(r"\s+", " ", " ".join(tokens)).strip()
        return cleaned

    def _summarize_if_needed(self, text: str) -> str:
        """Summarize lengthy prompts using the HF pipeline."""
        token_count = len(word_tokenize(text))
        if self.summarizer and token_count > self.summary_threshold:
            try:
                summary = self.summarizer(
                    text,
                    max_length=60,
                    min_length=15,
                    do_sample=False,
                )[0]["summary_text"]
                return summary
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"Summarization failed, using original text: {exc}")
        return text

    def can_translate(self, src_lang: str, target_lang: str) -> bool:
        return (
            self.translation_model is not None
            and self.translation_tokenizer is not None
            and self._normalize_lang(src_lang) in SUPPORTED_MBART_LANGS
            and self._normalize_lang(target_lang) in SUPPORTED_MBART_LANGS
        )

    def translate(self, text: str, src_lang: str, target_lang: str) -> Optional[str]:
        """Translate text via MBART if languages are supported."""
        if not text or not self.can_translate(src_lang, target_lang):
            return None
        src = SUPPORTED_MBART_LANGS[self._normalize_lang(src_lang)]
        tgt = SUPPORTED_MBART_LANGS[self._normalize_lang(target_lang)]
        tokenizer = self.translation_tokenizer
        model = self.translation_model
        assert tokenizer is not None and model is not None

        tokenizer.src_lang = src
        encoded = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt],
            max_new_tokens=256,
        )
        decoded = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        return decoded

    @staticmethod
    def _normalize_lang(lang: str) -> str:
        lang = (lang or "en").lower()
        return lang.split("-")[0]


# Reusable singleton when the module is imported multiple times
@lru_cache(maxsize=1)
def get_nlp_processor() -> Optional[NLPProcessor]:
    try:
        return NLPProcessor()
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"Failed to initialize NLPProcessor: {exc}")
        return None


