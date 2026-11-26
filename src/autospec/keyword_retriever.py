"""BM25 lexical specification retrieval with automotive-tuned tokenization and safe JSON persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Plus

from autospec.chunker import SpecificationChunk

# Automotive domain abbreviation expansions applied during query tokenization
AUTOMOTIVE_SYNONYMS: dict[str, list[str]] = {
    "atf": ["automatic", "transmission", "fluid"],
    "cvt": ["continuously", "variable", "transmission"],
    "abs": ["anti-lock", "brake", "system"],
    "cca": ["cold", "cranking", "amps"],
    "cyl": ["cylinder"],
    "diff": ["differential"],
    "ecu": ["engine", "control", "unit"],
    "egr": ["exhaust", "gas", "recirculation"],
    "maf": ["mass", "airflow"],
    "map": ["manifold", "absolute", "pressure"],
    "pcv": ["positive", "crankcase", "ventilation"],
    "tps": ["throttle", "position", "sensor"],
    "o2": ["oxygen", "sensor"],
    "ft-lb": ["lb-ft"],
    "ft-lbs": ["lb-ft"],
    "in-lb": ["lb-in"],
    "in-lbs": ["lb-in"],
    "n-m": ["nm"],
    "specs": ["specifications"],
    "spec": ["specification"],
    "cap": ["capacity"],
    "qty": ["quantity"],
    "seq": ["sequence"],
    "min": ["minimum"],
    "max": ["maximum"],
    "dia": ["diameter"],
    "temp": ["temperature"],
    "batt": ["battery"],
    "alt": ["alternator"],
    "rad": ["radiator"],
}


class AutomotiveTokenizer:
    """Specialized tokenizer preserving automotive part numbers, units, and fastener codes."""

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z0-9]+(?:[\-_][A-Za-z0-9]+)*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?",
        re.IGNORECASE,
    )

    STOPWORDS = {
        "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "on", "at",
        "by", "from", "is", "are", "was", "were", "be", "this", "that", "it", "what",
    }

    @classmethod
    def tokenize(cls, text: str, expand_synonyms: bool = False) -> list[str]:
        """Convert input text into normalized automotive-aware search tokens."""
        if not text:
            return []

        raw_tokens = cls.TOKEN_PATTERN.findall(text)
        tokens: list[str] = []

        for tok in raw_tokens:
            low = tok.lower()
            if low not in cls.STOPWORDS:
                tokens.append(low)
                if "-" in low:
                    for sub in low.split("-"):
                        if sub and sub not in cls.STOPWORDS and len(sub) > 1:
                            tokens.append(sub)
                if expand_synonyms and low in AUTOMOTIVE_SYNONYMS:
                    for syn in AUTOMOTIVE_SYNONYMS[low]:
                        if syn not in tokens:
                            tokens.append(syn)

        return tokens


class BM25KeywordRetriever:
    """Lexical specification search engine using BM25Plus with safe JSON serialization."""

    def __init__(self) -> None:
        self.bm25: BM25Plus | None = None
        self.chunks: list[SpecificationChunk] = []
        self.tokenized_corpus: list[list[str]] = []

    def build_index(self, chunks: list[SpecificationChunk]) -> None:
        """Tokenize chunks and build the BM25 inverted index."""
        if not chunks:
            raise ValueError("Cannot build BM25 index with 0 specification chunks.")

        self.chunks = chunks
        self.tokenized_corpus = []

        for ch in chunks:
            vehicle_str = " ".join(
                str(x) for x in (ch.vehicle_year, ch.vehicle_make, ch.vehicle_model) if x
            )
            full_chunk_text = (
                f"{ch.content} {ch.component} {ch.table_name} {ch.section_title} "
                f"{ch.subsystem} {ch.raw_evidence} {ch.part_number} {ch.conditions} {vehicle_str}"
            )
            tokens = AutomotiveTokenizer.tokenize(full_chunk_text, expand_synonyms=False)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Plus(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        subsystem_filter: str | None = None,
        document_ids: set[str] | None = None,
        manufacturer_filter: str | None = None,
        model_filter: str | None = None,
        year_filter: int | None = None,
    ) -> list[tuple[SpecificationChunk, float]]:
        """Search BM25 index for exact component and specification token matches."""
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        query_tokens = AutomotiveTokenizer.tokenize(query, expand_synonyms=True)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[tuple[SpecificationChunk, float]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            chunk_tokens = set(self.tokenized_corpus[idx])
            if not query_token_set.intersection(chunk_tokens):
                continue

            chunk = self.chunks[idx]
            if not self._matches_filters(
                chunk,
                subsystem_filter=subsystem_filter,
                document_ids=document_ids,
                manufacturer_filter=manufacturer_filter,
                model_filter=model_filter,
                year_filter=year_filter,
            ):
                continue

            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _matches_filters(
        chunk: SpecificationChunk,
        subsystem_filter: str | None = None,
        document_ids: set[str] | None = None,
        manufacturer_filter: str | None = None,
        model_filter: str | None = None,
        year_filter: int | None = None,
    ) -> bool:
        if document_ids is not None and chunk.document_id not in document_ids:
            return False
        if subsystem_filter and chunk.subsystem != subsystem_filter:
            return False
        if manufacturer_filter and manufacturer_filter.lower() not in (chunk.vehicle_make or "").lower():
            return False
        if model_filter and model_filter.lower() not in (chunk.vehicle_model or "").lower():
            return False
        if year_filter is not None and chunk.vehicle_year != year_filter:
            return False
        return True

    def save(self, file_path: Path) -> None:
        """Atomically serialize tokenized BM25 corpus as safe JSON (no pickle)."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        payload: dict[str, Any] = {
            "format": "autospec_bm25_json_v2",
            "tokenized_corpus": self.tokenized_corpus,
        }
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_path.replace(file_path)

    def load(self, file_path: Path, chunks: list[SpecificationChunk]) -> bool:
        """Safely load BM25 corpus from JSON and reconstruct BM25Plus."""
        if not file_path.exists():
            return False

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            corpus = data.get("tokenized_corpus")
            if not isinstance(corpus, list) or len(corpus) != len(chunks):
                return False
            self.tokenized_corpus = corpus
            self.chunks = chunks
            self.bm25 = BM25Plus(self.tokenized_corpus)
            return True
        except Exception:
            # Never fall back to pickle.load for untrusted index files; trigger clean rebuild instead
            return False
