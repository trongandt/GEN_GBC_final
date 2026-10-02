"""GEN-CIM locality-biased cosine KNN 1-opt moves, without objective calls.

Retains quota K//|S|+1 followed by global cosine ranking and cap K.
This quota DOES NOT guarantee that every selected node survives the cap.
Ties are deterministic (similarity descending, removed ID, added ID).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class NeighborCandidate:
    seed_set: object
    removed: int
    added: int
    similarity: float
    rank: int = 0


def generate_neighbors_with_meta(S, h_v: Tensor, k: int = 5) -> list[NeighborCandidate]:
    from phase2_trajectory import SeedSet
    if k < 0 or h_v.ndim != 2:
        raise ValueError("Need K>=0 and h_v[N,d]")
    ids = sorted(S.nodes)
    if ids and (ids[0] < 0 or ids[-1] >= len(h_v)):
        raise ValueError("Seed IDs outside h_v")
    if not k or not ids or len(ids) == len(h_v):
        return []
    if not bool(torch.isfinite(h_v).all()):
        raise ValueError("Nonfinite embeddings")
    pool = [v for v in range(len(h_v)) if v not in S.nodes]
    norm = F.normalize(h_v.detach(), p=2, dim=1, eps=1e-8)
    pool_ids = torch.tensor(pool, dtype=torch.long, device=h_v.device)
    # Query every selected node in one matrix multiply.
    sims = norm[ids] @ norm[pool_ids].T
    quota = min(k // len(ids) + 1, len(pool))
    raw = []
    for i, removed in enumerate(ids):
        # pool is ascending IDs; stable sorting breaks cosine ties by ID.
        top = torch.argsort(sims[i], descending=True, stable=True)[:quota]
        for j in top.cpu().tolist():
            raw.append((float(sims[i, j]), removed, pool[j]))
    raw.sort(key=lambda row: (-row[0], row[1], row[2]))
    seen, out = set(), []
    for similarity, removed, added in raw:
        nodes = (S.nodes - {removed}) | {added}
        if nodes in seen:
            continue
        seen.add(nodes)
        out.append(NeighborCandidate(SeedSet(nodes), removed, added, similarity, len(out)))
        if len(out) == k:
            break
    return out


def generate_neighbors(S, h_v: Tensor, k: int = 5) -> list:
    return [c.seed_set for c in generate_neighbors_with_meta(S, h_v, k)]


def top_candidate(S, h_v: Tensor, k: int = 5):
    result = generate_neighbors(S, h_v, k)
    return result[0] if result else None


def generate_neighbors_ranked_by_score(S, h_v: Tensor, score_fn, k: int = 5):
    return sorted([(s, float(score_fn(s))) for s in generate_neighbors(S, h_v, k)],
                  key=lambda pair: (-pair[1], tuple(sorted(pair[0].nodes))))
