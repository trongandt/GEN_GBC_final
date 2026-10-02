"""GEN-GBC Phase 2A -> B -> C+D -> B+ -> elite, runnable on Phase 1 outputs.

ExactGBCScorer supplies all ground truth. Midpoints retain raw ValueNet soft
labels even when the same set also occurs as an endpoint, as in GEN-CIM. Repeated source
rows stay separate, preserving their original endpoint/midpoint weights.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
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
            verified = endpoint or not use_proxy_for_midpoints
            score = float(scorer.cache[key]) if verified else float(step.score)
            # Keep midpoint soft labels and weights exactly as in GEN-CIM.
            step.score, step.is_exact_verified = score, verified
            samples.append(DataSample(step.S, score,
                                      endpoint_weight if endpoint else midpoint_weight,
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
    # Original adaptive gate, with the user-required STRICT qualification.
    quality_floor: float = 1.
    quality_ratio: float = .5
    top_trajectories: int = 50
    endpoint_weight: float = 1.
    midpoint_weight: float = .3
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
    crossover_pairs: int = 3
    crossover_n: int = 3
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
        counts = (self.H, self.k_neighbors, self.top_trajectories, self.perturb_top_k,
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


def augment_endpoint_neighborhood(samples, graph, h_v: Tensor, scorer, config: Phase2Config):
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
    if not parents:
        print("[Phase2B+] Không có endpoint exact để mở rộng.", flush=True)
        return samples, []
    rng = random.Random(cfg.seed + 9999)
    norms = h_v.detach().norm(dim=-1).cpu().tolist()
    existing = {s.seed_set.nodes for s in samples}
    seen = set(existing)
    proposals = []
    def propose(nodes, threshold, weight, name):
        S = SeedSet(nodes)
        validate_seed_set(S, graph.num_nodes, cfg.k)
        if S.nodes in seen:
            return
        seen.add(S.nodes)
        proposals.append((S, threshold, weight, name))
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
    if cfg.crossover:
        pairs = list(itertools.islice(itertools.combinations(range(len(parents)), 2),
                                     cfg.crossover_pairs))
        for i, j in pairs:
            a, b = parents[i], parents[j]
            union = sorted(a.seed_set.nodes | b.seed_set.nodes)
            threshold = cfg.perturb_threshold*max(a.score, b.score)
            for _ in range(cfg.crossover_n):
                propose(rng.sample(union, cfg.k), threshold, 1., f'crossover_ep{i}xep{j}')
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
    next_idx = max((s.trajectory_idx for s in samples), default=-1)+1
    elite = [DataSample(s.seed_set, s.score, w, True, next_idx+i, 0, f'elite_{s.strategy}')
             for i, (s, w) in enumerate(zip(top, weights))]
    print(f"[Elite] {len(elite)} unique exact sets; weights={weights}.", flush=True)
    return samples+elite, elite


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
    timings: dict = field(default_factory=dict)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(samples=[_sample_record(s) for s in self.samples],
                       exact_cache=[dict(nodes=sorted(key), score=sc) for key, sc in self.cache.items()],
                       pair_cache=[dict(nodes=sorted(key), score=sc) for key, sc in self.pair_cache.items()],
                       h_G=self.h_G.detach().cpu(), graph_fingerprint=self.graph_fingerprint,
                       score_kind='exact_raw_ordered_pair_internal_node_GBC',
                       config=asdict(self.config), Ffloor=self.Ffloor,
                       initial_experts={name: sorted(s.nodes) for name, s in self.initial_experts.items()},
                       kept_experts=list(self.kept_experts), expert_scores=self.expert_scores,
                       trajectories=[dict(strategy=name, steps=[dict(nodes=sorted(s.S.nodes),
                                     score=s.score, is_exact_verified=s.is_exact_verified) for s in traj])
                                     for name, traj in zip(self.trajectory_dataset.strategy_names,
                                                           self.trajectory_dataset.trajectories)],
                       timings=self.timings)
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
    """No Phase 1 changes needed: pass run_phase1(..., return_labels=True).

    Orchestration follows inspected GEN-CIM source: bootstrap,
    surrogate-only trajectories, exact endpoints, retrain, B+, elite.
    Both B+ branches are ON by default per the user's GEN-GBC configuration.
    B+/elite rows feed Phase 3; the source does NOT retrain ValueNet after B+.
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
    pair_scorer = (scorer if cfg.k == 2 else
                   ExactGBCScorer(graph, scorer.graph_path, scorer.binary, 2, cfg.exact_threads)
                   if graph.num_nodes >= 2 else None)
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
    initial = init_seed_sets(graph, cfg.k, h_v, bc_raw, pair_scorer=pair_scorer,
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
    print(f"[Phase2A] 42->{len(kept)} sources; Ffloor={floor:.6f}; "
          "5 compulsory sources bypass gate; others require >=Ffloor.", flush=True)
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
    boot_x = compute_seed_embedding_batch(h_v, list(kept.values()))
    boot_y = torch.tensor(list(kept_labels.values()), dtype=torch.float64, device=dev)
    t = time.perf_counter()
    trainer.fit(boot_x, boot_y, h_G_batch=h_G if cfg.use_context else None)
    times['bootstrap_training_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    trajectories = build_all_trajectories(graph, cfg.k, h_v, model, seed_sets_override=kept,
                                          H=cfg.H, k_neighbors=cfg.k_neighbors, h_G=h_G,
                                          early_stop=cfg.early_stop,
                                          endpoint_weight=cfg.endpoint_weight,
                                          midpoint_weight=cfg.midpoint_weight)
    times['surrogate_trajectory_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    samples, cache = build_dataset_from_trajectory_dataset(graph, trajectories, scorer)
    times['exact_stage_c_dataset_seconds'] = time.perf_counter()-t
    x, y, w = samples_to_tensors(samples, h_v)
    trainer.config.epochs = cfg.retrain_epochs
    t = time.perf_counter()
    trainer.fit(x, y, w, h_G_batch=h_G if cfg.use_context else None)
    times['retraining_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    samples, _ = augment_endpoint_neighborhood(samples, graph, h_v, scorer, cfg)
    samples, _ = add_elite_buffer(samples, cfg)
    times['augmentation_and_elite_seconds'] = time.perf_counter()-t
    times['total_phase2_seconds'] = time.perf_counter()-start
    fingerprint = hashlib.sha256(json.dumps(dict(n=graph.num_nodes, directed=graph.directed,
                       arcs=graph.edge_index.detach().cpu().t().tolist()),
                       separators=(',', ':')).encode()).hexdigest()
    result = Phase2Result(samples, cache, h_G, model, trainer, trajectories, initial, kept,
                          all_scores, floor, pair_scorer.cache if pair_scorer else {}, cfg,
                          fingerprint, times)
    if checkpoint_dir is not None:
        result.save(checkpoint_dir)
        torch.save(bc_raw.detach().cpu(), Path(checkpoint_dir)/'bc_raw.pt')
    print(f"[Phase2] Xong sau {times['total_phase2_seconds']:.1f}s; "
          f"best exact={best_sample(samples).score:.6f}; samples={len(samples)}.", flush=True)
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
    parser.add_argument('--centra-nodes', help='JSON ARRAY of INTERNAL IDs from SAME graph (optional)')
    args = parser.parse_args()
    from graph_utils import load_edge_list
    graph, _ = load_edge_list(args.graph, directed=False)
    h_v = torch.load(args.embeddings, map_location='cpu', weights_only=True)
    bc = torch.load(args.bc_raw, map_location='cpu', weights_only=True) if args.bc_raw else None
    nodes = json.loads(Path(args.centra_nodes).read_text()) if args.centra_nodes else None
    cfg = Phase2Config(k=args.k, seed=args.seed, exact_threads=args.threads,
                        perturb_sampling=args.perturb, crossover=args.crossover,
                        quality_floor=args.quality_floor, quality_ratio=args.quality_ratio)
    run_phase2(graph, h_v, bc, config=cfg, graph_path=args.graph, centra_nodes=nodes,
               checkpoint_dir=args.output, device=args.device)


if __name__ == '__main__':
    _main()
