"""Verified GEN-GBC Phase 2 search, exact archive and unique generative data.

ValueNet ranks mixed proposals; exact scores accept moves. The exact training
archive includes rejected proposals while Dcand retains quality and diversity.
Trajectory edges are serialized separately from canonical candidate rows.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import random
import time
from typing import Optional

import torch
from torch import Tensor

from gbc_types import GraphData
from phase2_trajectory import (SeedSet, Trajectory, TrajectoryDataset, ALWAYS_KEEP,
                               _validate_graph, _raw_bc, validate_seed_set,
                               init_seed_sets, quality_filter, build_all_trajectories)
from value_net import (ValueNetwork, ValueNetTrainer, TrainConfig,
                       compute_seed_embedding_batch)
from candidate_archive import CandidateArchive


@dataclass(frozen=True)
class DataSample:
    seed_set: SeedSet
    score: float
    weight: float
    is_exact_verified: bool
    trajectory_idx: int
    step_idx: int
    strategy: str = ''

    def __post_init__(self):
        if (not math.isfinite(self.score) or not math.isfinite(self.weight) or
                self.weight <= 0 or (self.is_exact_verified and self.score < 0)):
            raise ValueError("Invalid sample score/weight")

    @property
    def is_mc_verified(self):
        """Compatibility with GEN-CIM consumers: here verification is EXACT."""
        return self.is_exact_verified


def build_dataset(graph, trajectories: list[Trajectory], scorer, strategy_names=None, *,
                  endpoint_weight: float = 1., midpoint_weight: float = .3,
                  use_proxy_for_midpoints: bool = True):
    names = strategy_names if strategy_names is not None else [''] * len(trajectories)
    if len(names) != len(trajectories) or endpoint_weight <= 0 or midpoint_weight <= 0:
        raise ValueError("Invalid trajectory labels/weights")
    to_verify = []
    for traj in trajectories:
        for i, step in enumerate(traj):
            validate_seed_set(step.S, graph.num_nodes, scorer.k)
            if i in (0, len(traj)-1) or not use_proxy_for_midpoints:
                to_verify.append(step.S)
    scorer.score_many(to_verify)
    samples = []
    for ti, (traj, name) in enumerate(zip(trajectories, names)):
        for i, step in enumerate(traj):
            endpoint = i in (0, len(traj)-1)
            key = frozenset(step.S.nodes)
            verified = key in scorer.cache
            score = float(scorer.cache[key]) if verified else float(step.score)
            # An available exact label takes precedence in every occurrence.
            step.score, step.is_exact_verified = score, verified
            samples.append(DataSample(step.S, score,
                                      endpoint_weight if verified else midpoint_weight,
                                      verified, ti, i, name))
    print(f"[Phase2C+D] D_traj={len(samples)} rows, "
          f"exact={sum(s.is_exact_verified for s in samples)}, "
          f"proxy={sum(not s.is_exact_verified for s in samples)}.", flush=True)
    return samples, scorer.cache


def build_dataset_from_trajectory_dataset(graph, traj_dataset: TrajectoryDataset, scorer, **kwargs):
    kwargs.setdefault('endpoint_weight', traj_dataset.endpoint_weight)
    kwargs.setdefault('midpoint_weight', traj_dataset.midpoint_weight)
    return build_dataset(graph, traj_dataset.trajectories, scorer,
                         traj_dataset.strategy_names, **kwargs)


def samples_to_tensors(samples: list[DataSample], h_v: Tensor):
    embeddings = compute_seed_embedding_batch(h_v, [s.seed_set for s in samples]).detach()
    scores = torch.tensor([s.score for s in samples], dtype=torch.float64, device=h_v.device)
    weights = torch.tensor([s.weight for s in samples], dtype=torch.float32, device=h_v.device)
    return embeddings, scores, weights


def filter_by_exact_verified(samples):
    return [s for s in samples if s.is_exact_verified]


def filter_by_mc_verified(samples):
    return filter_by_exact_verified(samples)


def best_sample(samples):
    return max(samples, key=lambda s: (s.is_exact_verified, s.score), default=None)


@dataclass
class Phase2Config:
    k: int = 10
    seed: int = 42
    H: int = 5
    k_neighbors: int = 5
    early_stop: bool = True
    verified_search: bool = True
    proposal_per_source: int = 2
    random_per_node: int = 1
    shortlist_size: int = 8
    exploration_slots: int = 2
    max_batches_per_step: int = 2
    trajectory_evaluation_budget: int = 64
    improvement_rtol: float = 1e-10
    improvement_atol: float = 1e-8
    online_every_trajectories: int = 3
    online_epochs: int = 50
    candidate_quality_ratio: float = .95
    candidate_max_size: int = 256
    candidate_min_distance: int = 1
    validation_ratio: float = .2
    # Original adaptive gate, with the user-required STRICT qualification.
    quality_floor: float = 1.
    quality_ratio: float = .5
    top_trajectories: int = 50
    endpoint_weight: float = 1.
    midpoint_weight: float = .3
    hedge_epsilon: float = .1
    hedge_seeds_per_budget: int = 3
    hedge_quality_ratio: float = .9
    bootstrap_epochs: int = 500
    retrain_epochs: int = 500
    patience: int = 50
    log_every: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    lambda_rank: float = .5
    rank_pairs_per_batch: int = 32
    use_context: bool = False
    hidden_dims: tuple[int, ...] = (256, 128)
    exact_threads: int = 2
    verify_raw_bc: bool = True
    # Both B+ branches run by default, as requested for GEN-GBC.
    perturb_sampling: bool = True
    perturb_top_k: int = 5
    perturb_r1_n: int = 10
    perturb_r2_n: int = 5
    perturb_r3_n: int = 2
    perturb_threshold: float = .85
    crossover: bool = True
    crossover_pairs: int = 6
    crossover_n: int = 20
    crossover_parent_top_k: int = 10
    crossover_quality_ratio: float = .85
    crossover_exhaustive_limit: int = 100
    crossover_max_proposals: int = 300
    # GEN-CIM's post-B+ elite buffer: explicit extra rows, NOT source weighting.
    elite_top_k: int = 5
    elite_weight: float = 4.
    elite_rank_weights: bool = True
    soft_weight_temp: float = 2.
    centra_epsilon: float = .1
    centra_delta: float = .05
    centra_trials: int = 100
    centra_initial_samples: int = 512

    def validate(self):
        new_counts = (self.proposal_per_source, self.random_per_node, self.shortlist_size,
                      self.exploration_slots, self.max_batches_per_step,
                      self.trajectory_evaluation_budget, self.online_every_trajectories,
                      self.online_epochs, self.candidate_max_size, self.candidate_min_distance)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in new_counts):
            raise ValueError('Search/archive counts must be nonnegative integers')
        if (self.proposal_per_source < 1 or self.shortlist_size < 1 or
                self.max_batches_per_step < 1 or self.candidate_max_size < 1 or
                not 0 <= self.exploration_slots < self.shortlist_size or
                not 0 <= self.candidate_quality_ratio <= 1 or
                not 0 <= self.validation_ratio < 1 or
                not math.isfinite(self.improvement_rtol) or self.improvement_rtol < 0 or
                not math.isfinite(self.improvement_atol) or self.improvement_atol < 0):
            raise ValueError('Invalid verified-search/archive configuration')
        if not 0 < self.hedge_epsilon < 1-1/math.e:
            raise ValueError("Invalid HEDGE epsilon")
        if self.hedge_seeds_per_budget < 1:
            raise ValueError("Need at least one HEDGE seed per budget")
        if not 0 <= self.hedge_quality_ratio <= 1 or not 0 <= self.crossover_quality_ratio <= 1:
            raise ValueError("Invalid diversity quality ratios")
        counts = (self.crossover_parent_top_k, self.crossover_exhaustive_limit,
                  self.crossover_max_proposals, self.H, self.k_neighbors, self.top_trajectories, self.perturb_top_k,
                  self.perturb_r1_n, self.perturb_r2_n, self.perturb_r3_n,
                  self.crossover_pairs, self.crossover_n, self.elite_top_k)
        if any(not isinstance(v, int) or v < 0 for v in counts):
            raise ValueError("Counts must be nonnegative integers")
        if (not 0 <= self.perturb_threshold <= 1 or self.endpoint_weight <= 0 or
                self.midpoint_weight <= 0 or self.elite_weight <= 0 or
                self.soft_weight_temp <= 0 or self.exact_threads < 1):
            raise ValueError("Invalid weights/threshold/threads")
        if (not math.isfinite(self.quality_floor) or self.quality_floor < 0 or
                not 0 <= self.quality_ratio <= 1):
            raise ValueError("Invalid quality gate")


def _top_unique_exact(samples, n):
    seen, top = set(), []
    for s in sorted(filter_by_exact_verified(samples),
                    key=lambda s: (-s.score, tuple(sorted(s.seed_set.nodes)),
                                   s.trajectory_idx, s.step_idx)):
        if s.seed_set.nodes not in seen:
            seen.add(s.seed_set.nodes)
            top.append(s)
        if len(top) >= n:
            break
    return top if n > 0 else []


def augment_endpoint_neighborhood(samples, graph, h_v: Tensor, scorer, config: Phase2Config, *, reference_set=None):
    """GEN-CIM B+: norm-paired r=1, random r=2/3, union-sample crossover.

    Node norms are the original heuristic, not a theorem about GBC gain.
    Exact score >= .85 * parent (or best parent) accepts a proposed variant.
    Identical parents and children do not consume multiple augmentation slots.
    A zero requested count produces zero variants (fix original max(1,0)).
    """
    cfg = config
    if not (cfg.perturb_sampling or cfg.crossover):
        print("[Phase2B+] Đã tắt bằng cấu hình.", flush=True)
        return samples, []
    print(f"[Phase2B+] Chạy: perturb={cfg.perturb_sampling}, "
          f"crossover={cfg.crossover}.", flush=True)
    parents = _top_unique_exact(samples, cfg.perturb_top_k)
    if not parents and not cfg.crossover:
        print("[Phase2B+] Không có endpoint exact để mở rộng.", flush=True)
        return samples, []
    rng = random.Random(cfg.seed + 9999)
    norms = h_v.detach().norm(dim=-1).cpu().tolist()
    existing = {s.seed_set.nodes for s in samples}
    seen = set(existing)
    proposals = []
    proposal_index = {}
    def propose(nodes, threshold, weight, name):
        S = SeedSet(nodes)
        validate_seed_set(S, graph.num_nodes, cfg.k)
        if S.nodes in proposal_index:
            idx = proposal_index[S.nodes]
            old = proposals[idx]
            if threshold < old[1]:
                proposals[idx] = (S, threshold, weight, name)
            return False
        if S.nodes in seen:
            return False
        seen.add(S.nodes)
        proposal_index[S.nodes] = len(proposals)
        proposals.append((S, threshold, weight, name))
        return True
    if cfg.perturb_sampling:
        for rank, ep in enumerate(parents):
            selected = sorted(ep.seed_set.nodes)
            available = [v for v in range(graph.num_nodes) if v not in ep.seed_set.nodes]
            ascending = sorted(selected, key=lambda v: (norms[v], v))
            descending = sorted(available, key=lambda v: (-norms[v], v))
            scale = max(.25, 1.-rank*.2)
            for radius, base_count, weight in ((1, cfg.perturb_r1_n, 1.),
                                              (2, cfg.perturb_r2_n, .9),
                                              (3, cfg.perturb_r3_n, .8)):
                if not base_count:
                    continue
                count = max(1, round(base_count*scale))
                eff = min(radius, cfg.k, len(available))
                if not eff:
                    continue
                for i in range(count):
                    if radius == 1:
                        j = i % min(len(ascending), len(descending))
                        nodes = (ep.seed_set.nodes-{ascending[j]}) | {descending[j]}
                    else:
                        nodes = ((ep.seed_set.nodes-set(rng.sample(selected, eff)))
                                 | set(rng.sample(available, eff)))
                    propose(nodes, cfg.perturb_threshold*ep.score, weight,
                            f'perturb_r{radius}_rank{rank}')
    if cfg.crossover and cfg.crossover_pairs and cfg.crossover_max_proposals:
        cross_parents = _top_unique_exact(samples, cfg.crossover_parent_top_k)
        if cross_parents:
            best_score = cross_parents[0].score
            cross_parents = [s for s in cross_parents if s.score >= cfg.crossover_quality_ratio*best_score]
            ref = next((s for s in samples if s.is_exact_verified and
                        s.seed_set == reference_set), None) if reference_set is not None else None
            if ref is not None and all(s.seed_set != ref.seed_set for s in cross_parents):
                cross_parents.append(ref)
            anchor = next((i for i, s in enumerate(cross_parents)
                           if s.seed_set == reference_set), 0)
            pairs = list(itertools.combinations(range(len(cross_parents)), 2))
            def pair_key(pair):
                a, b = (cross_parents[i] for i in pair)
                distance = cfg.k-len(a.seed_set.nodes & b.seed_set.nodes)
                return (-distance, -min(a.score, b.score), pair)
            anchored = sorted([p for p in pairs if anchor in p], key=pair_key)
            others = sorted([p for p in pairs if anchor not in p], key=pair_key)
            chosen = []
            while len(chosen) < cfg.crossover_pairs and (anchored or others):
                for pool in (anchored, others):
                    if pool and len(chosen) < cfg.crossover_pairs:
                        chosen.append(pool.pop(0))
            new_count = 0
            for i, j in chosen:
                if new_count >= cfg.crossover_max_proposals:
                    break
                a, b = cross_parents[i], cross_parents[j]
                union = sorted(a.seed_set.nodes | b.seed_set.nodes)
                combinations = math.comb(len(union), cfg.k)
                threshold = cfg.perturb_threshold*max(a.score, b.score)
                exhaustive = combinations <= cfg.crossover_exhaustive_limit
                name = f'crossover_{a.strategy}__{b.strategy}'
                print(f"[Crossover] {a.strategy} x {b.strategy}: union={len(union)}, "
                      f"combinations={combinations}, exhaustive={exhaustive}", flush=True)
                if exhaustive:
                    for nodes in itertools.combinations(union, cfg.k):
                        if new_count >= cfg.crossover_max_proposals:
                            break
                        new_count += int(propose(nodes, threshold, 1., name))
                else:
                    accepted = 0
                    for _ in range(max(1, cfg.crossover_n*100)):
                        if accepted >= cfg.crossover_n or new_count >= cfg.crossover_max_proposals:
                            break
                        fresh = propose(rng.sample(union, cfg.k), threshold, 1., name)
                        accepted += int(fresh)
                        new_count += int(fresh)
            print(f"[Crossover] selected_pairs={len(chosen)}, new_unique={new_count}, "
                  f"budget={cfg.crossover_max_proposals}", flush=True)
    values = scorer.score_many([S for S, _, _, _ in proposals])
    next_idx = max((s.trajectory_idx for s in samples), default=-1) + 1
    added = []
    for (S, threshold, weight, name), value in zip(proposals, values):
        if value >= threshold:
            added.append(DataSample(S, float(value), weight, True, next_idx+len(added), 0, name))
    print(f"[Phase2B+] {len(parents)} unique parents; {len(proposals)} unique proposals; "
          f"giữ {len(added)} qua ngưỡng {cfg.perturb_threshold:.2f} × parent.", flush=True)
    return samples+added, added


def add_elite_buffer(samples, config: Phase2Config):
    cfg = config
    top = _top_unique_exact(samples, cfg.elite_top_k)
    if not top:
        return samples, []
    if cfg.elite_rank_weights:
        weights = [cfg.elite_weight*(.5**i) for i in range(len(top))]
    else:
        # Algebraically equivalent to original exp/scale, stable for GBC millions.
        weights = [cfg.elite_weight*math.exp((s.score-top[0].score)/cfg.soft_weight_temp)
                   for s in top]
        weights = [max(1e-12, w) for w in weights]
    boosts = {s.seed_set.nodes: w for s, w in zip(top, weights)}
    # Upgrade a canonical row instead of appending duplicate elite examples.
    samples = [replace(s, weight=s.weight+boosts.get(s.seed_set.nodes, 0.)) for s in samples]
    elite = [s for s in samples if s.seed_set.nodes in boosts]
    print(f"[Elite] {len(elite)} unique exact sets; weights={weights}.", flush=True)
    return samples, elite


def canonicalize_samples(samples, cache):
    """One row per set; exact overrides proxy, duplicate weights never sum."""
    rows = {}
    for s in samples:
        key = s.seed_set.nodes
        if key in cache:
            s = replace(s, score=float(cache[key]), is_exact_verified=True,
                        weight=max(1., s.weight))
        old = rows.get(key)
        if old is None or (s.is_exact_verified, s.weight) > (old.is_exact_verified, old.weight):
            rows[key] = s
    return list(rows.values())


def archive_samples(archive):
    return [DataSample(r.seed_set, r.exact_score, 1., True, -1, -1,
                       '|'.join(sorted(r.sources))) for r in archive.exact_records]


def fit_archive(trainer, archive, h_v, h_G, cfg, *, epochs, stage):
    """Stable set-level holdout excluded from bootstrap and every later fit.

    Search adaptively uses exact scores, so this diagnostic is not an
    independent final test set. Prequential audits complement it.
    """
    rows = archive_samples(archive)
    if not rows:
        return dict(stage=stage, train_sets=0, validation_sets=0)
    if not hasattr(trainer, '_archive_training_anchor'):
        trainer._archive_training_anchor = min(tuple(sorted(s.seed_set.nodes)) for s in rows)
    def held_out(s):
        key = tuple(sorted(s.seed_set.nodes))
        digest = hashlib.sha256((str(cfg.seed)+':'+','.join(map(str, key))).encode()).hexdigest()
        return key != trainer._archive_training_anchor and int(digest, 16)/2**256 < cfg.validation_ratio
    validation = [s for s in rows if held_out(s)]
    training = [s for s in rows if not held_out(s)]
    x, y, w = samples_to_tensors(training, h_v)
    trainer.config.epochs = epochs
    trainer.fit(x, y, w, h_G_batch=h_G if cfg.use_context else None)
    report = dict(stage=stage, train_sets=len(training), validation_sets=len(validation),
                  validation_scope='never_fitted_but_adaptive_search_not_independent_test',
                  validation_nodes=[sorted(s.seed_set.nodes) for s in validation])
    if validation:
        vx, vy, _ = samples_to_tensors(validation, h_v)
        report.update(trainer.evaluate(vx, vy, h_G if cfg.use_context else None))
    return report


def _sample_record(sample):
    return dict(nodes=sorted(sample.seed_set.nodes), score=sample.score, weight=sample.weight,
                is_exact_verified=sample.is_exact_verified, trajectory_idx=sample.trajectory_idx,
                step_idx=sample.step_idx, strategy=sample.strategy)


@dataclass
class Phase2Result:
    samples: list[DataSample]
    cache: dict
    h_G: Tensor
    value_net: ValueNetwork
    trainer: ValueNetTrainer
    trajectory_dataset: TrajectoryDataset
    initial_experts: dict
    kept_experts: dict
    expert_scores: dict
    Ffloor: float
    pair_cache: dict
    config: Phase2Config
    graph_fingerprint: str
    initialization_audit: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    archive: CandidateArchive = field(default_factory=CandidateArchive)
    value_net_metrics: list[dict] = field(default_factory=list)

    @property
    def best_exact(self):
        return self.archive.best

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        row_index = {s.seed_set.nodes: i for i, s in enumerate(self.samples)}
        edges = sorted({(row_index[a.S.nodes], row_index[b.S.nodes])
                        for traj in self.trajectory_dataset.trajectories
                        for a, b in zip(traj.steps, traj.steps[1:])
                        if a.S.nodes in row_index and b.S.nodes in row_index})
        payload = dict(schema_version=2, samples=[_sample_record(s) for s in self.samples],
                       trajectory_edges=[list(edge) for edge in edges],
                       candidate_archive=self.archive.to_records(),
                       best_exact=(dict(nodes=sorted(self.best_exact.seed_set.nodes),
                                        score=self.best_exact.exact_score) if self.best_exact else None),
                       value_net_metrics=self.value_net_metrics,
                       exact_cache=[dict(nodes=sorted(key), score=sc) for key, sc in self.cache.items()],
                       pair_cache=[dict(nodes=sorted(key), score=sc) for key, sc in self.pair_cache.items()],
                       h_G=self.h_G.detach().cpu(), graph_fingerprint=self.graph_fingerprint,
                       score_kind='exact_raw_ordered_pair_internal_node_GBC',
                       config=asdict(self.config), Ffloor=self.Ffloor,
                       initial_experts={name: sorted(s.nodes) for name, s in self.initial_experts.items()},
                       kept_experts=list(self.kept_experts), expert_scores=self.expert_scores,
                       trajectories=[dict(strategy=name, stop_reason=traj.stop_reason,
                                     audit=traj.audit, steps=[dict(nodes=sorted(s.S.nodes),
                                     score=s.score, is_exact_verified=s.is_exact_verified) for s in traj])
                                     for name, traj in zip(self.trajectory_dataset.strategy_names,
                                                           self.trajectory_dataset.trajectories)],
                       initialization_audit=self.initialization_audit, timings=self.timings)
        torch.save(payload, directory/'phase2_dataset.pt')
        self.trainer.save(directory/'phase2_value_net.pt')
        audit = {key: value for key, value in payload.items() if key not in ('h_G', 'trajectories')}
        audit['counts'] = dict(initial_sources=len(self.initial_experts),
                               unique_initial_sets=len(set(self.initial_experts.values())),
                               kept_sources=len(self.kept_experts), samples=len(self.samples),
                               unique_sample_sets=len({s.seed_set for s in self.samples}))
        (directory/'phase2_summary.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print(f"[Phase2] Đã lưu dataset, ValueNet và summary: {directory}", flush=True)


def run_phase2(graph: GraphData, h_v: Tensor, bc_raw: Optional[Tensor] = None, *,
               config: Optional[Phase2Config] = None, scorer=None, graph_path=None,
               exact_source=None, exact_binary=None, centra_source=None, centra_binary=None,
               centra_nodes=None, checkpoint_dir=None, device=None) -> Phase2Result:
    """Pass run_phase1(..., return_labels=True); no Phase 1 changes required.

    Verified search is the default. All exact observations train ValueNet;
    only quality/diversity-selected unique candidates enter generative data.
    """
    from exact_gbc_scorer import ExactGBCScorer, compile_exact_gbc
    cfg = config or Phase2Config()
    cfg.validate()
    _validate_graph(graph, cfg.k)
    if h_v.ndim != 2 or len(h_v) != graph.num_nodes or not bool(torch.isfinite(h_v).all()):
        raise ValueError("Need finite Phase 1 h_v[N,d]")
    dev = torch.device(device or h_v.device)
    h_v = h_v.detach().to(dev, torch.float32)
    h_G = h_v.mean(0)
    start, times = time.perf_counter(), {}
    if bc_raw is None:
        from phase1_representation import compute_node_betweenness_labels
        bc_raw = compute_node_betweenness_labels(graph)
    _raw_bc(bc_raw, graph.num_nodes)
    if scorer is None:
        if graph_path is None:
            raise ValueError("Supply graph_path to bind C++ to the SAME graph and ID map")
        source = Path(exact_source or Path(__file__).with_name('exact_gbc.cpp'))
        binary = compile_exact_gbc(source, exact_binary or source.with_name('exact_gbc_phase2'))
        scorer = ExactGBCScorer(graph, graph_path, binary, cfg.k, cfg.exact_threads)
    if (scorer.k != cfg.k or scorer.graph.num_nodes != graph.num_nodes or
            scorer.graph.directed != graph.directed or
            not torch.equal(scorer.graph.edge_index.cpu(), graph.edge_index.cpu())):
        raise ValueError("Exact scorer is bound to a different graph or budget")
    if graph.num_nodes == 1:
        # The supplied C++ requires n>=2. The empty ordered-pair domain has
        # an exact closed-form score of zero; seed the existing scorer cache.
        if float(bc_raw[0]) != 0.:
            raise ValueError("A one-node graph has raw singleton GBC=0")
        scorer.cache[frozenset({0})] = 0.
    if cfg.verify_raw_bc:
        singleton = scorer if cfg.k == 1 else ExactGBCScorer(
            graph, scorer.graph_path, scorer.binary, 1, cfg.exact_threads)
        ids = sorted(set([int(bc_raw.argmax()), int(bc_raw.argmin()), graph.num_nodes//2]))
        vals = singleton.score_many([SeedSet({v}) for v in ids])
        if any(not math.isclose(float(bc_raw[v]), sc, rel_tol=1e-8, abs_tol=1e-7)
               for v, sc in zip(ids, vals)):
            raise ValueError("bc_raw differs from exact singleton scores; check normalization/ID mapping")
        print(f"[Phase2A] BC raw checked against exact on {len(ids)} singleton groups.", flush=True)
    t = time.perf_counter()
    initialization_audit = {}
    initial = init_seed_sets(graph, cfg.k, h_v, bc_raw, hedge_epsilon=cfg.hedge_epsilon,
                            objective_scorer=scorer, audit=initialization_audit,
                            hedge_seeds_per_budget=cfg.hedge_seeds_per_budget,
                            hedge_quality_ratio=cfg.hedge_quality_ratio,
                            centra_nodes=centra_nodes, seed=cfg.seed, centra_source=centra_source,
                            centra_binary=centra_binary,
                            centra_options=dict(epsilon=cfg.centra_epsilon, delta=cfg.centra_delta,
                                                trials=cfg.centra_trials,
                                                initial_samples=cfg.centra_initial_samples))
    labels = scorer.score_many(list(initial.values()))
    all_scores = dict(zip(initial, labels))
    for name, sc in all_scores.items():
        print(f"[Phase2A] {name}: exact_raw_GBC={sc:.6f}; "
              f"{'always_keep' if name in ALWAYS_KEEP else 'quality_gate'}", flush=True)
    kept, kept_labels, floor = quality_filter(initial, labels, quality_floor=cfg.quality_floor,
                                             quality_ratio=cfg.quality_ratio,
                                             top_trajectories=cfg.top_trajectories)
    print(f"[Phase2A] 15->{len(kept)} sources; Ffloor={floor:.6f}; "
          "6 compulsory sources bypass gate; others require >=Ffloor.", flush=True)
    times['expert_generation_and_exact_stage_a_seconds'] = time.perf_counter()-t
    # Local initialization seed without resetting caller's global RNG state.
    with torch.random.fork_rng(devices=[dev.index or 0] if dev.type == 'cuda' else []):
        torch.manual_seed(cfg.seed)
        model = ValueNetwork(h_v.size(1), cfg.hidden_dims, cfg.use_context)
    train_cfg = TrainConfig(lr=cfg.lr, weight_decay=cfg.weight_decay,
                            epochs=cfg.bootstrap_epochs, batch_size=cfg.batch_size,
                            patience=cfg.patience, log_every=cfg.log_every,
                            lambda_rank=cfg.lambda_rank,
                            rank_pairs_per_batch=cfg.rank_pairs_per_batch, seed=cfg.seed)
    trainer = ValueNetTrainer(model, train_cfg, dev)
    t = time.perf_counter()
    # Keep all paid-for exact labels, including HEDGE trials not chosen as experts.
    archive = CandidateArchive()
    for name, S in initial.items():
        archive.add(S, all_scores[name], exact=True, source=name)
    for row in initialization_audit.get('hedge_candidates', []):
        archive.add(SeedSet(row['nodes']), row['exact_score'], exact=True,
                    source='hedge_trial_'+str(row['seed']))
    metrics = [fit_archive(trainer, archive, h_v, h_G, cfg,
                           epochs=cfg.bootstrap_epochs, stage='bootstrap')]
    print(f"[Phase2A] ValueNet bootstrap: {metrics[-1]['train_sets']} unique training sets.", flush=True)
    times['bootstrap_training_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    def update_from_search(index, dataset):
        for name, traj in zip(dataset.strategy_names, dataset.trajectories):
            for step in list(traj)+traj.evaluations:
                if step.S.nodes in scorer.cache:
                    archive.add(step.S, scorer.cache[step.S.nodes], exact=True, source=name)
        if (cfg.online_every_trajectories and cfg.online_epochs and
                index % cfg.online_every_trajectories == 0):
            metrics.append(fit_archive(trainer, archive, h_v, h_G, cfg,
                                       epochs=cfg.online_epochs, stage=f'online_{index}'))
    search_options = dict(bc_raw=bc_raw,
        expert_nodes=sorted(set().union(*(s.nodes for s in initial.values()))),
        per_source=cfg.proposal_per_source, random_per_node=cfg.random_per_node,
        shortlist_size=cfg.shortlist_size, exploration_slots=cfg.exploration_slots,
        max_batches_per_step=cfg.max_batches_per_step,
        evaluation_budget=cfg.trajectory_evaluation_budget, seed=cfg.seed,
        improvement_rtol=cfg.improvement_rtol, improvement_atol=cfg.improvement_atol)
    trajectories = build_all_trajectories(graph, cfg.k, h_v, model, seed_sets_override=kept,
                                          H=cfg.H, k_neighbors=cfg.k_neighbors, h_G=h_G,
                                          early_stop=cfg.early_stop,
                                          endpoint_weight=cfg.endpoint_weight,
                                          midpoint_weight=cfg.midpoint_weight,
                                          scorer=scorer if cfg.verified_search else None,
                                          verified_options=search_options,
                                          update_callback=update_from_search)
    times['trajectory_and_online_training_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    samples, cache = build_dataset_from_trajectory_dataset(graph, trajectories, scorer)
    # Endpoints verified in the legacy path also enter the exact archive.
    for s in samples:
        archive.add(s.seed_set, s.score, exact=s.is_exact_verified,
                    source=s.strategy, confidence_weight=s.weight)
    samples = canonicalize_samples(samples+archive_samples(archive), cache)
    times['exact_stage_c_dataset_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    metrics.append(fit_archive(trainer, archive, h_v, h_G, cfg,
                               epochs=cfg.retrain_epochs, stage='after_trajectories'))
    times['retraining_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    samples, _ = augment_endpoint_neighborhood(
        samples, graph, h_v, scorer, cfg, reference_set=initial['centra'])
    # Retain every B+ evaluation for ValueNet, including rejected proposals.
    # Existing sample rows supply precise provenance; cache-only rows are rejects.
    for s in samples:
        archive.add(s.seed_set, s.score, exact=s.is_exact_verified,
                    source=s.strategy, confidence_weight=s.weight)
    for key, score in cache.items():
        archive.add(SeedSet(key), score, exact=True,
                    source='bplus_evaluated' if key not in archive.records else '')
    selected = archive.select(quality_ratio=cfg.candidate_quality_ratio,
                              max_size=cfg.candidate_max_size,
                              min_distance=cfg.candidate_min_distance)
    # Distinct singleton IDs prevent old consumers grouping all canonical rows
    # into one artificial trajectory. New consumers should use trajectory_edges.
    samples = [DataSample(r.seed_set, r.exact_score, cfg.endpoint_weight, True, i, 0,
                          '|'.join(sorted(r.sources))) for i, r in enumerate(selected)]
    samples, _ = add_elite_buffer(samples, cfg)
    metrics.append(fit_archive(trainer, archive, h_v, h_G, cfg,
                               epochs=cfg.retrain_epochs, stage='after_bplus'))
    times['augmentation_and_elite_seconds'] = time.perf_counter()-t
    times['total_phase2_seconds'] = time.perf_counter()-start
    fingerprint = hashlib.sha256(json.dumps(dict(n=graph.num_nodes, directed=graph.directed,
                       arcs=graph.edge_index.detach().cpu().t().tolist()),
                       separators=(',', ':')).encode()).hexdigest()
    result = Phase2Result(samples, cache, h_G, model, trainer, trajectories, initial, kept,
                          all_scores, floor, {}, cfg,
                          fingerprint, initialization_audit=initialization_audit, timings=times,
                          archive=archive, value_net_metrics=metrics)
    if archive.best.exact_score < max(all_scores.values()):
        raise AssertionError('Best expert was lost from exact archive')
    audits = [row for traj in trajectories.trajectories for row in traj.audit]
    diagnostics = dict(stage='prequential_search', evaluated_moves=len(audits),
        predicted_improvements=sum(row['predicted_delta'] > 0 for row in audits),
        false_improvements=sum(row['predicted_delta'] > 0 and not row['exact_improves'] for row in audits))
    metrics.append(diagnostics)
    if checkpoint_dir is not None:
        result.save(checkpoint_dir)
        torch.save(bc_raw.detach().cpu(), Path(checkpoint_dir)/'bc_raw.pt')
    print(f"[Phase2] Xong sau {times['total_phase2_seconds']:.1f}s; "
          f"best exact={archive.best.exact_score:.6f}; unique Dcand={len(samples)}; "
          f"exact archive={len(archive.exact_records)}.", flush=True)
    return result


phase2 = run_phase2


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True, help='Same unweighted edge list as Phase 1')
    parser.add_argument('--embeddings', required=True, help='h_v .pt tensor from Phase 1')
    parser.add_argument('--bc-raw', help='Raw float64 BC .pt tensor; else recompute Brandes')
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default='experiments/checkpoints/phase2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--perturb', action=argparse.BooleanOptionalAction, default=True,
                        help='B+ perturb is ON by default; --no-perturb disables it')
    parser.add_argument('--crossover', action=argparse.BooleanOptionalAction, default=True,
                        help='B+ crossover is ON by default; --no-crossover disables it')
    parser.add_argument('--quality-floor', type=float, default=1.)
    parser.add_argument('--quality-ratio', type=float, default=.5)
    parser.add_argument('--verified-search', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--shortlist-size', type=int, default=8)
    parser.add_argument('--exploration-slots', type=int, default=2)
    parser.add_argument('--trajectory-evaluation-budget', type=int, default=64)
    parser.add_argument('--candidate-quality-ratio', type=float, default=.95)
    parser.add_argument('--candidate-max-size', type=int, default=256)
    parser.add_argument('--hedge-epsilon', type=float, default=.1)
    parser.add_argument('--centra-nodes', help='JSON ARRAY of INTERNAL IDs from SAME graph (optional)')
    args = parser.parse_args()
    from graph_utils import load_edge_list
    graph, _ = load_edge_list(args.graph, directed=False)
    h_v = torch.load(args.embeddings, map_location='cpu', weights_only=True)
    bc = torch.load(args.bc_raw, map_location='cpu', weights_only=True) if args.bc_raw else None
    nodes = json.loads(Path(args.centra_nodes).read_text()) if args.centra_nodes else None
    cfg = Phase2Config(k=args.k, seed=args.seed, exact_threads=args.threads,
                        verified_search=args.verified_search,
                        shortlist_size=args.shortlist_size, exploration_slots=args.exploration_slots,
                        trajectory_evaluation_budget=args.trajectory_evaluation_budget,
                        candidate_quality_ratio=args.candidate_quality_ratio,
                        candidate_max_size=args.candidate_max_size,
                        hedge_epsilon=args.hedge_epsilon,
                        perturb_sampling=args.perturb, crossover=args.crossover,
                        quality_floor=args.quality_floor, quality_ratio=args.quality_ratio)
    run_phase2(graph, h_v, bc, config=cfg, graph_path=args.graph, centra_nodes=nodes,
               checkpoint_dir=args.output, device=args.device)


if __name__ == '__main__':
    _main()
