"""BM25 lexical specification retrieval with automotive-tuned tokenization."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Plus

from autospec.chunker import SpecificationChunk


class AutomotiveTokenizer:
    """Specialized tokenizer preserving automotive part numbers, units, and fastener codes."""

    # Tokenizes words, alphanumeric codes (e.g., WSS-M6C62-A, PM-1-C, 206-03), and decimal numbers
    TOKEN_PATTERN = re.compile(
        r"[A-Za-z0-9]+(?:[\-_][A-Za-z0-9]+)*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?",
        re.IGNORECASE,
    )

    STOPWORDS = {
        "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "on", "at",
        "by", "from", "is", "are", "was", "were", "be", "this", "that", "it", "what",
    }

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Convert input text into normalized automotive-aware search tokens."""
        if not text:
            return []

        raw_tokens = cls.TOKEN_PATTERN.findall(text)
        tokens: list[str] = []

        for tok in raw_tokens:
            low = tok.lower()
            if low not in cls.STOPWORDS:
                tokens.append(low)
                # If token contains a hyphen (e.g. '206-03' or 'wss-m6c62-a'), also include sub-parts
                if "-" in low:
                    for sub in low.split("-"):
                        if sub and sub not in cls.STOPWORDS and len(sub) > 1:
                            tokens.append(sub)

        return tokens


class BM25KeywordRetriever:
    """Lexical specification search engine using BM25Plus for guaranteed positive, discriminative scoring."""

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
            # Combine content, component name, section, subsystem, and raw evidence
            full_chunk_text = (
                f"{ch.content} {ch.component} {ch.table_name} {ch.section_title} "
                f"{ch.subsystem} {ch.raw_evidence} {ch.part_number}"
            )
            tokens = AutomotiveTokenizer.tokenize(full_chunk_text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Plus(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        subsystem_filter: str | None = None,
    ) -> list[tuple[SpecificationChunk, float]]:
        """Search BM25 index for exact component and specification token matches."""
        if self.bm25 is None or len(self.chunks) == 0:
            return []

        query_tokens = AutomotiveTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        # Rank indices by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[tuple[SpecificationChunk, float]] = []
        for idx in ranked_indices:
            score = float(scores[idx])

            # Must share at least one lexical token with the query
            chunk_tokens = set(self.tokenized_corpus[idx])
            if not query_token_set.intersection(chunk_tokens):
                continue

            chunk = self.chunks[idx]
            if subsystem_filter and chunk.subsystem != subsystem_filter:
                continue

            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

    def save(self, file_path: Path) -> None:
        """Serialize BM25 index and tokens to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            pickle.dump(
                {
                    "tokenized_corpus": self.tokenized_corpus,
                    "bm25": self.bm25,
                },
                f,
            )

    def load(self, file_path: Path, chunks: list[SpecificationChunk]) -> bool:
        """Load BM25 index from disk and re-attach chunks."""
        if not file_path.exists():
            return False

        try:
            with file_path.open("rb") as f:
                data = pickle.load(f)
                self.tokenized_corpus = data["tokenized_corpus"]
                self.bm25 = data["bm25"]
                self.chunks = chunks
                return True
        except Exception:
            return False
