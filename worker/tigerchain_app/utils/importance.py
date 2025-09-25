from __future__ import annotations

from typing import Dict, Iterable


class DocumentImportanceScorer:
    """Heuristically scores documents and chunks to support reranking."""

    def __init__(
        self,
        *,
        base_score: float = 0.45,
        size_weight: float = 0.25,
        category_weight: float = 0.15,
        subject_weight: float = 0.15,
    ) -> None:
        self.base_score = base_score
        self.size_weight = size_weight
        self.category_weight = category_weight
        self.subject_weight = subject_weight

    def score_document(
        self,
        *,
        file_size_bytes: int,
        categories: Iterable[str],
        metadata: Dict[str, object] | None,
        subject_tags: Iterable[str],
    ) -> float:
        score = self.base_score
        size_factor = min(file_size_bytes / 1_000_000, 1.0)
        score += size_factor * self.size_weight

        normalised_categories = {str(cat).strip().lower() for cat in categories if str(cat).strip()}
        if normalised_categories & {"policy", "legal", "compliance"}:
            score += self.category_weight
        elif normalised_categories:
            score += self.category_weight * 0.5

        subject_set = {tag.strip().lower() for tag in subject_tags if tag}
        if subject_set & {"compliance", "security"}:
            score += self.subject_weight
        elif subject_set & {"product", "engineering"}:
            score += self.subject_weight * 0.7
        elif subject_set:
            score += self.subject_weight * 0.4

        priority_hint = None
        if metadata:
            priority_hint = metadata.get("priority") or metadata.get("importance")
        if isinstance(priority_hint, str):
            value = priority_hint.strip().lower()
            if value in {"critical", "high"}:
                score += 0.2
            elif value in {"medium", "default"}:
                score += 0.1
        elif isinstance(priority_hint, (int, float)):
            score += min(max(float(priority_hint), 0.0), 1.0) * 0.2

        return round(max(0.0, min(score, 1.0)), 3)

    def score_chunk(
        self,
        *,
        chunk_length: int,
        document_score: float,
        subject_tags: Iterable[str],
    ) -> float:
        density = min(chunk_length / 900, 1.0)
        subject_set = {tag.strip().lower() for tag in subject_tags if tag}
        subject_boost = 0.15 if subject_set & {"compliance", "security"} else 0.0
        score = (document_score * 0.6) + (density * 0.25) + subject_boost
        return round(max(0.0, min(score, 1.0)), 3)

    def rank_subjects(self, subject_tags: Iterable[str], document_score: float) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for index, tag in enumerate(sorted({tag.strip().lower() for tag in subject_tags if tag})):
            decay = max(0.35, 1.0 - (index * 0.2))
            weights[tag] = round(document_score * decay, 3)
        return weights


__all__ = ["DocumentImportanceScorer"]
