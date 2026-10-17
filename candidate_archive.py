"""Canonical exact labels and provenance, independent of trajectory edges."""
from dataclasses import dataclass, field
import math


@dataclass
class CandidateRecord:
    seed_set: object
    exact_score: float | None = None
    proxy_score: float | None = None
    confidence_weight: float = 1.
    sources: set[str] = field(default_factory=set)

    @property
    def score(self):
        return self.exact_score if self.exact_score is not None else self.proxy_score


class CandidateArchive:
    def __init__(self):
        self.records = {}

    def add(self, S, score, *, exact, source='', confidence_weight=1.):
        score = float(score)
        if (not math.isfinite(score) or (exact and score < 0) or
                not math.isfinite(confidence_weight) or confidence_weight <= 0):
            raise ValueError('Invalid candidate label/weight')
        record = self.records.setdefault(S.nodes, CandidateRecord(S))
        if exact:
            if record.exact_score is not None and not math.isclose(
                    record.exact_score, score, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError('Conflicting exact labels for one candidate')
            record.exact_score = score
            # Verification upgrades label confidence independently of row count.
            record.confidence_weight = 1.
        else:
            record.proxy_score = score
            if record.exact_score is None:
                record.confidence_weight = confidence_weight
        if source:
            record.sources.add(source)
        return record

    @property
    def exact_records(self):
        return sorted((r for r in self.records.values() if r.exact_score is not None),
                      key=lambda r: (-r.exact_score, tuple(sorted(r.seed_set.nodes))))

    @property
    def best(self):
        return next(iter(self.exact_records), None)

    def select(self, *, quality_ratio=.95, max_size=256, min_distance=1):
        """Quality gate then farthest-first node diversity; always retain best.

        This is node diversity only, not an assertion of path diversity. A
        sparse archive is returned as-is rather than padded with duplicates.
        """
        if not 0 <= quality_ratio <= 1 or max_size < 1 or min_distance < 0:
            raise ValueError('Invalid candidate selection settings')
        rows = self.exact_records
        if not rows:
            return []
        remaining = [r for r in rows[1:] if r.exact_score >= quality_ratio*rows[0].exact_score]
        selected = [rows[0]]
        while remaining and len(selected) < max_size:
            def distance(row):
                return min(len(row.seed_set.nodes - s.seed_set.nodes) for s in selected)
            chosen = min(remaining, key=lambda r:
                         (-distance(r), -r.exact_score, tuple(sorted(r.seed_set.nodes))))
            if distance(chosen) < min_distance:
                break
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def to_records(self):
        return [dict(nodes=sorted(r.seed_set.nodes), exact_score=r.exact_score,
                     proxy_score=r.proxy_score, confidence_weight=r.confidence_weight,
                     sources=sorted(r.sources))
                for r in sorted(self.records.values(),
                                key=lambda r: tuple(sorted(r.seed_set.nodes)))]
