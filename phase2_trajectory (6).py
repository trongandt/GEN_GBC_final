"""GEN-GBC Phase 2A's 15 source slots and GEN-CIM Phase 2B trajectories.

All IDs are Phase 1 internal IDs. No community/diffusion/SPAGAN dependency.
ExactGBCScorer stays in its existing module; import it lazily to avoid its
existing import of SeedSet causing a circular import.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
import json
import math
import random
import subprocess
import tempfile
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from gbc_types import GraphData

ALWAYS_KEEP = frozenset({'degree', 'centra', 'top_bc', 'bridging', 'hedge_q1_10', 'hedge_q2_10'})


@dataclass(frozen=True)
class SeedSet:
    nodes: frozenset[int]

    def __post_init__(self):
        values = list(self.nodes)
        if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in values):
            raise ValueError("Seed nodes must be nonnegative integer internal IDs")
        if len(values) != len(set(values)):
            raise ValueError("Repeated node in SeedSet input")
        object.__setattr__(self, 'nodes', frozenset(int(v) for v in values))

    @property
    def k(self):
        return len(self.nodes)

    def to_tensor(self, device=None):
        return torch.tensor(sorted(self.nodes), dtype=torch.long, device=device)

    def to_sorted_list(self):
        return sorted(self.nodes)

    def to_binary_tensor(self, num_nodes: int, device='cpu'):
        if self.nodes and max(self.nodes) >= num_nodes:
            raise ValueError("Seed ID outside binary vector")
        vector = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        vector[self.to_tensor(device)] = 1.
        return vector

    @classmethod
    def from_binary_tensor(cls, tensor: Tensor, threshold: float = .5):
        if tensor.ndim != 1:
            raise ValueError("Need a one-dimensional binary/probability vector")
        return cls((tensor >= threshold).nonzero(as_tuple=True)[0].cpu().tolist())

    def __contains__(self, node):
        return node in self.nodes

    def __iter__(self):
        return iter(sorted(self.nodes))

    def __len__(self):
        return self.k

    def __repr__(self):
        return f"SeedSet(k={self.k}, nodes={sorted(self.nodes)})"


@dataclass
class TrajectoryStep:
    S: SeedSet
    score: float
    is_exact_verified: bool = False


@dataclass
class Trajectory:
    steps: list[TrajectoryStep] = field(default_factory=list)

    def append(self, step):
        self.steps.append(step)

    @property
    def initial(self):
        return self.steps[0]

    @property
    def terminal(self):
        return self.steps[-1]

    @property
    def length(self):
        return len(self.steps)

    @property
    def best(self):
        return max(self.steps, key=lambda step: step.score)

    @property
    def improvement(self):
        return self.terminal.score-self.initial.score if len(self.steps) > 1 else 0.

    def scores(self):
        return [step.score for step in self.steps]

    def __getitem__(self, idx):
        return self.steps[idx]

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return self.length


@dataclass
class TrajectoryDataset:
    trajectories: list[Trajectory] = field(default_factory=list)
    strategy_names: list[str] = field(default_factory=list)
    endpoint_weight: float = 1.0
    midpoint_weight: float = 0.3

    def add(self, traj, name):
        self.trajectories.append(traj)
        self.strategy_names.append(name)

    @property
    def num_trajectories(self):
        return len(self.trajectories)

    def all_steps_with_weights(self):
        for traj in self.trajectories:
            for i, step in enumerate(traj):
                yield step, (self.endpoint_weight if i in (0, traj.length-1)
                             else self.midpoint_weight)

    def total_steps(self):
        return sum(t.length for t in self.trajectories)

    def best_seed_set(self):
        steps = [step for traj in self.trajectories for step in traj]
        return max(steps, key=lambda step: step.score).S if steps else None


def _validate_graph(graph: GraphData, k: int):
    if not isinstance(k, Integral) or isinstance(k, bool) or not 1 <= k <= graph.num_nodes:
        raise ValueError("Require 1 <= k <= |V|")
    if graph.directed or graph.is_weighted:
        raise ValueError("The agreed 15-expert recipe is for UNDIRECTED UNWEIGHTED graphs")
    arcs = graph.edge_index.detach().cpu().t().tolist()
    keys = {(u, v) for u, v in arcs}
    if (len(keys) != len(arcs) or any(u == v for u, v in keys) or
            any((v, u) not in keys for u, v in keys)):
        raise ValueError("Use Phase 1's simple bidirectional COO graph")


def validate_seed_set(S: SeedSet, n: int, k: int):
    if S.k != k or min(S.nodes, default=-1) < 0 or max(S.nodes, default=n) >= n:
        raise ValueError(f"Need exactly {k} distinct IDs in [0,{n})")


def _raw_bc(bc_raw: Tensor, n: int) -> list[float]:
    if not isinstance(bc_raw, Tensor) or bc_raw.shape != (n,) or bc_raw.dtype != torch.float64:
        raise ValueError("Use Phase 1 raw BC CPU float64 vector; not log/normalized/float32 labels")
    vals = bc_raw.detach().cpu().tolist()
    if any(not math.isfinite(v) or v < 0 for v in vals):
        raise ValueError("Raw BC must be finite and nonnegative")
    return vals


def _bc_order(bc, ids=None):
    return sorted(range(len(bc)) if ids is None else ids, key=lambda v: (-bc[v], v))


def _adjacency(graph):
    adj = [set() for _ in range(graph.num_nodes)]
    for u, v in graph.edge_index.detach().cpu().t().tolist():
        adj[u].add(v)
    return [sorted(neighbors) for neighbors in adj]


def _fill(nodes, k, n, rng):
    """Called only when positive BC pool cannot fill the budget."""
    selected = set(nodes)
    if len(selected) < k:
        selected.update(rng.sample([v for v in range(n) if v not in selected], k-len(selected)))
    return SeedSet(selected)


def _candidate_pool(bc, k):
    positive = _bc_order(bc, [v for v, score in enumerate(bc) if score > 0])
    pool = positive[:min(5*k, len(positive))]
    if len(pool) < k:
        # Positive nodes first; deterministic zero-BC fillers for fixed experts.
        pool += [v for v in _bc_order(bc) if v not in set(pool)][:k-len(pool)]
    return pool


def init_degree(graph: GraphData, k: int) -> SeedSet:
    _validate_graph(graph, k)
    adj = _adjacency(graph)
    return SeedSet(sorted(range(graph.num_nodes), key=lambda v: (-len(adj[v]), v))[:k])


def init_greedy_embedding(graph, k: int, h_v: Tensor, bc_raw: Tensor) -> SeedSet:
    _validate_graph(graph, k)
    bc = _raw_bc(bc_raw, graph.num_nodes)
    if h_v.ndim != 2 or len(h_v) != graph.num_nodes or not bool(torch.isfinite(h_v).all()):
        raise ValueError("Need finite Phase 1 embeddings h_v[N,d]")
    if k == graph.num_nodes:
        return SeedSet(range(k))
    pool = _candidate_pool(bc, k)
    norm = F.normalize(h_v.detach().cpu(), p=2, dim=1, eps=1e-8)
    selected = [pool[0]]
    while len(selected) < k:
        remaining = [v for v in pool if v not in selected]
        # D(v,S)=min_u (1-cos(v,u)) = 1-max_u cos(v,u).
        distances = 1 - (norm[remaining] @ norm[selected].T).max(dim=1).values
        best = min(range(len(remaining)),
                   key=lambda i: (-float(distances[i]), -bc[remaining[i]], remaining[i]))
        selected.append(remaining[best])
    return SeedSet(selected)


def init_bridging(graph, k: int, bc_raw: Tensor) -> SeedSet:
    _validate_graph(graph, k)
    bc = _raw_bc(bc_raw, graph.num_nodes)
    adj = _adjacency(graph)
    deg = [len(row) for row in adj]
    scores = [bc[v] * (1./deg[v]) / sum(1./deg[u] for u in adj[v])
              if deg[v] else 0. for v in range(graph.num_nodes)]
    return SeedSet(sorted(range(graph.num_nodes), key=lambda v: (-scores[v], v))[:k])


def run_centra(graph, k: int, bc_raw: Tensor, *, source=None, binary=None,
               seed: int = 42, epsilon: float = 0.1, delta: float = 0.05,
               trials: int = 100, initial_samples: int = 512) -> SeedSet:
    """Compile/run the repo's CentRA once; emit IDs directly in model ID space.

    The C++ loader drops self-loop-only isolates. Exporting canonical MODEL
    IDs as external labels avoids its local sorted-map shifting returned IDs.
    Isolates do not affect the selected greedy coverage solution. Its stopping
    certificate is on this active graph; GEN-GBC labels are always exact on
    the full graph. All-zero BC and k>=active_nodes are solved trivially.
    """
    _validate_graph(graph, k)
    bc = _raw_bc(bc_raw, graph.num_nodes)
    if not (0 < epsilon < 1-1/math.e and 0 < delta < 1 and trials >= 1 and initial_samples >= 1):
        raise ValueError("Invalid CentRA sampling parameters")
    adj = _adjacency(graph)
    active = [v for v, row in enumerate(adj) if row]
    if k == graph.num_nodes or not any(bc) or k >= len(active):
        print("[CentRA] Degenerate/saturated budget: exact coverage optimum is trivial.", flush=True)
        return SeedSet(_bc_order(bc)[:k]) if not any(bc) else _fill(active, k, graph.num_nodes, random.Random(seed))
    source = Path(source or Path(__file__).with_name('centra.cpp')).resolve()
    binary = Path(binary or source.with_name('centra_phase2')).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Need repo CentRA source: {source}")
    if not binary.is_file() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        binary.parent.mkdir(parents=True, exist_ok=True)
        print("[CentRA] Biên dịch...", flush=True)
        subprocess.run(['g++', '-O2', '-std=c++17', str(source), '-o', str(binary)], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        edges = Path(tmp)/'model_ids.txt'
        edges.write_text(''.join(f'{u} {v}\n' for u, row in enumerate(adj) for v in row if u < v))
        cmd = [str(binary), '--graph', str(edges), '--k', str(k), '--seed', str(seed),
               '--epsilon', str(epsilon), '--delta', str(delta), '--trials', str(trials),
               '--initial-samples', str(initial_samples)]
        print(f"[CentRA] Chạy 1 nghiệm: k={k}, seed={seed}; log tiến độ từ C++.", flush=True)
        # Inherit stderr so progress is visible while CentRA is running.
        payload = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout)
    result = SeedSet(payload['nodes'])
    validate_seed_set(result, graph.num_nodes, k)
    if (payload.get('method') != 'CentRA' or payload.get('k') != k or
            payload.get('num_nodes') != len(active) or payload.get('directed') is not False or
            payload.get('status') != 'stopping_condition_met'):
        raise ValueError("Unexpected CentRA result or stopping status")
    return result


def run_hedge(graph, k, bc_raw, *, numerator, denominator=10, seed=42, epsilon=.1, source=None, binary=None):
    """Run HEDGE with ceil(numerator*q/denominator), q=ceil(k*ln(n)/epsilon^2).

    q uses the full model graph's n. The exported active graph omits isolates;
    these are filled deterministically if k exceeds its size. Exact labels
    are always evaluated on the full original graph by the caller.
    """
    _validate_graph(graph, k)
    if denominator < 1 or numerator not in (1, 2) or not 0 < epsilon < 1-1/math.e:
        raise ValueError("Invalid HEDGE fraction/epsilon")
    adj = _adjacency(graph)
    active = [v for v, row in enumerate(adj) if row]
    if not any(_raw_bc(bc_raw, graph.num_nodes)):
        return SeedSet(range(k))
    if k >= len(active):
        return _fill(active, k, graph.num_nodes, random.Random(seed))
    q = math.ceil(k * math.log(graph.num_nodes) / epsilon**2)
    samples = (numerator*q + denominator-1)//denominator
    source = Path(source or Path(__file__).with_name('hedge.cpp')).resolve()
    binary = Path(binary or source.with_name('hedge_phase2')).resolve()
    if not binary.is_file() or binary.stat().st_mtime_ns < source.stat().st_mtime_ns:
        binary.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['g++', '-O2', '-std=c++17', str(source), '-o', str(binary)], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        edges = Path(tmp)/'model_ids.txt'
        edges.write_text(''.join(f'{u} {v}\n' for u, row in enumerate(adj) for v in row if u < v))
        cmd = [str(binary), '--graph', str(edges), '--k', str(k), '--seed', str(seed),
               '--epsilon', str(epsilon), '--samples', str(samples)]
        payload = json.loads(subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True).stdout)
    result = SeedSet(payload['nodes'])
    validate_seed_set(result, graph.num_nodes, k)
    if (payload.get('method') != 'HEDGE' or payload.get('samples') != samples or
            payload.get('num_nodes') != len(active) or payload.get('directed') is not False):
        raise ValueError("Unexpected HEDGE result")
    print(f"[HEDGE] fraction={numerator}/{denominator}, q={q}, samples={samples}, seed={seed}", flush=True)
    return result


def init_seed_sets(graph, k: int, h_v: Tensor, bc_raw: Tensor, *,
                   pair_scorer=None, centra_nodes=None, seed: int = 42,
                   centra_source=None, centra_binary=None, centra_options=None, hedge_epsilon=.1,
                   hedge_source=None, hedge_binary=None, objective_scorer=None,
                   hedge_seeds_per_budget=3, hedge_quality_ratio=.9, audit=None) -> dict[str, SeedSet]:
    """Exactly 15 source slots before Stage A gate, retaining duplicate sources."""
    _validate_graph(graph, k)
    bc = _raw_bc(bc_raw, graph.num_nodes)
    rng, n = random.Random(seed), graph.num_nodes
    positive = [v for v, score in enumerate(bc) if score > 0]
    names = {}
    names['degree'] = init_degree(graph, k)
    names['greedy_embedding'] = init_greedy_embedding(graph, k, h_v, bc_raw)
    names['centra'] = (SeedSet(centra_nodes) if centra_nodes is not None else
                       run_centra(graph, k, bc_raw, source=centra_source, binary=centra_binary,
                                  seed=seed, **(centra_options or {})))
    names['top_bc'] = SeedSet(_bc_order(bc)[:k])
    names['bridging'] = init_bridging(graph, k, bc_raw)
    if objective_scorer is None:
        raise ValueError("Exact objective_scorer is required for multi-seed HEDGE selection")
    if hedge_seeds_per_budget < 1 or not 0 <= hedge_quality_ratio <= 1:
        raise ValueError("Invalid multi-seed HEDGE selection settings")
    reference = names['centra']
    reference_score = objective_scorer.score_many([reference])[0]
    audit = audit if audit is not None else {}
    audit['hedge_candidates'] = []
    selected_hedges = []
    for numerator in (1, 2):
        candidates = []
        for trial in range(hedge_seeds_per_budget):
            trial_seed = seed + 1000*numerator + trial
            S = run_hedge(graph, k, bc_raw, numerator=numerator, denominator=10,
                          seed=trial_seed, epsilon=hedge_epsilon,
                          source=hedge_source, binary=hedge_binary)
            candidates.append((trial_seed, S))
        scores = objective_scorer.score_many([S for _, S in candidates])
        rows = [(trial_seed, S, float(score))
                for (trial_seed, S), score in zip(candidates, scores)]
        qualified = [r for r in rows if r[2] >= hedge_quality_ratio*reference_score]
        fallback = not qualified
        pool = qualified or rows
        # Prefer a distinct qualifying group when one exists. Quality fallback
        # chooses the best score, not an arbitrarily weak diverse group.
        distinct = [r for r in pool if r[1] not in selected_hedges]
        pool = distinct or pool
        anchors = [reference] + selected_hedges
        def diversity(S):
            return min(k-len(S.nodes & a.nodes) for a in anchors)
        chosen = (min(pool, key=lambda r: (-r[2], -diversity(r[1]), r[0])) if fallback
                  else min(pool, key=lambda r: (-diversity(r[1]), -r[2], r[0])))
        chosen_seed, chosen_set, chosen_score = chosen
        name = f'hedge_q{numerator}_10'
        names[name] = chosen_set
        selected_hedges.append(chosen_set)
        for trial_seed, S, score in rows:
            audit['hedge_candidates'].append(dict(source=name, seed=trial_seed,
                nodes=sorted(S.nodes), exact_score=score,
                distance_to_centra=k-len(S.nodes & reference.nodes),
                qualified=score >= hedge_quality_ratio*reference_score,
                selected=trial_seed == chosen_seed, quality_fallback=fallback))
        print(f"[HEDGE select] {name}: seed={chosen_seed}, exact={chosen_score:.6f}, "
              f"distance_to_centra={k-len(chosen_set.nodes & reference.nodes)}, "
              f"quality_fallback={fallback}", flush=True)
    # Four ordinary r=2 variants and four r=4 variants whose ADDED nodes
    # are outside the union of CentRA and the two selected HEDGE groups.
    expert_union = reference.nodes | selected_hedges[0].nodes | selected_hedges[1].nodes
    seen = {S.nodes for S in names.values()}
    audit['semi_good'] = []
    for parent_name in ('centra', 'hedge_q1_10', 'hedge_q2_10', 'top_bc'):
        parent = names[parent_name]
        for requested in (2, 4):
            external = requested == 4
            forbidden = (expert_union | parent.nodes) if external else parent.nodes
            outside = [v for v in range(n) if v not in forbidden]
            radius = min(requested, k, len(outside))
            candidate = parent
            for attempt in range(100):
                removed = rng.sample(sorted(parent.nodes), radius)
                pool = [v for v in outside if bc[v] > 0]
                added = []
                for _ in range(min(radius, len(pool))):
                    v = rng.choices(pool, weights=[math.sqrt(bc[u]) for u in pool], k=1)[0]
                    added.append(v)
                    pool.remove(v)
                if len(added) < radius:
                    added.extend(rng.sample([v for v in outside if v not in added], radius-len(added)))
                candidate = SeedSet((parent.nodes-set(removed)) | set(added))
                if candidate.nodes not in seen or radius == 0:
                    break
            name = f'semi_good_{parent_name}_r{requested}' + ('_external' if external else '')
            duplicate = candidate.nodes in seen
            names[name] = candidate
            seen.add(candidate.nodes)
            audit['semi_good'].append(dict(source=name, requested_radius=requested,
                effective_radius=radius, external=external, duplicate=duplicate,
                added_nodes=sorted(candidate.nodes-parent.nodes)))
            print(f"[SemiGood] {name}: radius={radius}, duplicate={duplicate}", flush=True)
    for S in names.values():
        validate_seed_set(S, n, k)
    assert len(names) == 15
    print(f"[Phase2A] 15 nguồn sinh; {len(set(names.values()))} tập node duy nhất; "
          f"|V+|={len(positive)}.", flush=True)
    return names


def quality_filter(seed_sets, scores, *, quality_floor: float = 1., quality_ratio: float = .5,
                   top_trajectories: int = 50):
    """Keep six compulsory sources and qualified remaining sources.

    GEN-CIM adaptive Ffloor=max(floor,ratio*best_filterable) and count cap.
    No min-random rescue: such a rescue would violate the user's hard gate.
    Equality qualifies (>=), as in the original source.
    """
    if (len(scores) != len(seed_sets) or quality_floor < 0 or
            not math.isfinite(quality_floor) or not 0 <= quality_ratio <= 1 or top_trajectories < 0):
        raise ValueError("Invalid Stage A filter configuration")
    rows = [(name, S, float(score)) for (name, S), score in zip(seed_sets.items(), scores)]
    if any(not math.isfinite(sc) or sc < 0 for _, _, sc in rows):
        raise ValueError("Invalid exact Stage A labels")
    compulsory = [row for row in rows if row[0] in ALWAYS_KEEP]
    flexible = sorted([row for row in rows if row[0] not in ALWAYS_KEEP],
                      key=lambda row: (-row[2], row[0]))
    floor = max(quality_floor, quality_ratio * max((sc for _, _, sc in flexible), default=0.))
    eligible = [row for row in flexible if row[2] >= floor]
    if top_trajectories:
        eligible = eligible[:max(0, top_trajectories-len(compulsory))]
    kept = compulsory + eligible
    return {name: S for name, S, _ in kept}, {name: sc for name, _, sc in kept}, floor


def build_trajectory(S_init, h_v: Tensor, V_phi, H: int = 5,
                     k_neighbors: int = 5, h_G: Optional[Tensor] = None,
                     early_stop: bool = True) -> Trajectory:
    from neighbor import generate_neighbors
    if H < 0 or k_neighbors < 0:
        raise ValueError("H and K must be nonnegative")
    traj = Trajectory([TrajectoryStep(S_init, V_phi.predict(h_v, S_init, h_G))])
    for _ in range(H):
        neighbors = generate_neighbors(traj.terminal.S, h_v, k_neighbors)
        if not neighbors:
            break
        scores = V_phi.predict_batch(h_v, neighbors, h_G)
        best = min(range(len(scores)), key=lambda i: (-scores[i], tuple(sorted(neighbors[i].nodes))))
        if early_stop and scores[best] <= traj.terminal.score:
            break
        traj.append(TrajectoryStep(neighbors[best], float(scores[best])))
    return traj


def build_all_trajectories(graph, k: int, h_v: Tensor, V_phi, *,
                           seed_sets_override: dict, H: int = 5, k_neighbors: int = 5,
                           h_G=None, early_stop=True, endpoint_weight=1., midpoint_weight=.3):
    _validate_graph(graph, k)
    dataset = TrajectoryDataset(endpoint_weight=endpoint_weight, midpoint_weight=midpoint_weight)
    for i, (name, S) in enumerate(seed_sets_override.items(), 1):
        validate_seed_set(S, graph.num_nodes, k)
        traj = build_trajectory(S, h_v, V_phi, H, k_neighbors, h_G, early_stop)
        dataset.add(traj, name)
        print(f"[Phase2B] {i}/{len(seed_sets_override)} {name}: steps={traj.length}, "
              f"proxy_raw={traj.initial.score:.6f}->{traj.terminal.score:.6f}", flush=True)
    return dataset
