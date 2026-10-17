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
    source: str = 'cosine'


def propose_mixed_neighbors(S, h_v: Tensor, *, bc_raw=None, expert_nodes=(),
                            per_source: int = 2, random_per_node: int = 1,
                            seed: int = 42):
    """Offer each removed node cosine, expert, BC and exploratory additions.

    No global cosine cap: every seed gets the same proposal opportunities.
    Returned metadata survives deduplication and is used by the audit log.
    """
    import random
    from phase2_trajectory import SeedSet
    if per_source < 1 or random_per_node < 0:
        raise ValueError('Invalid mixed-neighbor quotas')
    ids = sorted(S.nodes)
    if h_v.ndim != 2 or not bool(torch.isfinite(h_v).all()):
        raise ValueError('Need finite h_v[N,d]')
    n = len(h_v)
    if ids and (ids[0] < 0 or ids[-1] >= n):
        raise ValueError('Seed outside embedding matrix')
    outside = [v for v in range(n) if v not in S.nodes]
    if not outside or not ids:
        return []
    bc = ([0.] * n if bc_raw is None else
          torch.as_tensor(bc_raw).detach().cpu().tolist())
    if len(bc) != n:
        raise ValueError('BC size mismatch')
    expert = sorted(set(expert_nodes) - S.nodes)
    if any(v < 0 or v >= n for v in expert):
        raise ValueError('Expert ID outside graph')
    ranked_bc = sorted(outside, key=lambda v: (-bc[v], v))
    ranked_expert = sorted(expert, key=lambda v: (-bc[v], v))
    norm = F.normalize(h_v.detach(), p=2, dim=1, eps=1e-8)
    sims = (norm[ids] @ norm[outside].T).cpu()
    rng, seen, result = random.Random(seed), set(), []
    for i, removed in enumerate(ids):
        nearest = torch.argsort(sims[i], descending=True, stable=True).tolist()
        choices = [('cosine', [outside[j] for j in nearest[:per_source]]),
                   ('expert', ranked_expert[:per_source]),
                   ('bc', ranked_bc[:per_source]),
                   ('random', rng.sample(outside, min(random_per_node, len(outside))))]
        for source, additions in choices:
            for added in additions:
                nodes = (S.nodes - {removed}) | {added}
                if nodes in seen:
                    continue
                seen.add(nodes)
                similarity = float((norm[removed] * norm[added]).sum())
                result.append(NeighborCandidate(SeedSet(nodes), removed, added,
                                               similarity, len(result), source))
    return result


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
