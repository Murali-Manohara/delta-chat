"""
Retrieval over PID A, PID B, and the delta report.

Every retrievable unit is a `Chunk` with a stable `source_ref` that the
answer layer can turn directly into a citation: `pidA:page3:blk_abc123`
or `delta:mod_abc123_def456`. This is what makes "grounding" concrete
instead of aspirational -- an answer is grounded if and only if every
claim in it traces to a `Chunk.source_ref` that was actually retrieved.

Retrieval method: TF-IDF + cosine similarity (scikit-learn). This is a
deliberate choice over a full embedding model:
  - The corpus here is small (hundreds of short blocks + a delta report),
    where TF-IDF's sparse lexical matching is *better* for this domain
    than dense embeddings: instrument tags ("26-PIT-9055"), exact
    dimensions ("50 barg"), and page numbers are precisely the kind of
    high-specificity tokens embeddings tend to blur together, while
    TF-IDF weights them highly (high document frequency discrimination).
  - No network/API dependency for retrieval means `make chat` works
    offline / without an LLM key at all (see llm.py's extractive
    fallback) -- retrieval quality is decoupled from LLM availability.
  - Documented trade-off (see README "what we cut"): a production system
    would likely hybridize TF-IDF with embeddings for paraphrase-style
    questions ("what changed near the pump?" when no block literally
    contains the word "pump"). We note this as a concrete "what's next".
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.canonical.model import CanonicalDocument
from src.delta.engine import DeltaResult


@dataclass
class Chunk:
    source_ref: str      # e.g. "pidA:page3:a1b2c3d4e5f6" or "delta:mod_..."
    doc_label: str        # "PID A", "PID B", or "Delta Report"
    page: int | None
    text: str


class RetrievalIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add_document(self, doc: CanonicalDocument, label: str):
        for blk in doc.blocks:
            self.chunks.append(Chunk(
                source_ref=f"{label}:page{blk.page}:{blk.block_id}",
                doc_label=label,
                page=blk.page,
                text=blk.text,
            ))

    def add_delta_report(self, delta: DeltaResult):
        for e in delta.entries:
            text = f"[{e.kind.value}/{e.change_type}] {e.location}. "
            if e.before is not None:
                text += f"Before: {e.before}. "
            if e.after is not None:
                text += f"After: {e.after}. "
            self.chunks.append(Chunk(
                source_ref=f"delta:{e.id}",
                doc_label="Delta Report",
                page=e.page,
                text=text,
            ))

    def build(self):
        if not self.chunks:
            raise ValueError("RetrievalIndex.build() called with zero chunks")
        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if self._vectorizer is None:
            self.build()
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return [(self.chunks[i], float(sims[i])) for i in ranked[:top_k] if sims[i] > 0]
