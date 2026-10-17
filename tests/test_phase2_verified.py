"""Regression checks for search decisions, labels, budgets and persistence.

Run: python -m unittest discover -s tests -v
"""
import itertools
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from candidate_archive import CandidateArchive
from dataset_builder import (DataSample, Phase2Config, add_elite_buffer,
                             augment_endpoint_neighborhood, build_dataset, canonicalize_samples, fit_archive,
                             run_phase2)
from exact_gbc_scorer import ExactGBCScorer, compile_exact_gbc
from graph_utils import load_edge_list
from neighbor import propose_mixed_neighbors
from phase1_representation import compute_node_betweenness_labels
from phase2_trajectory import (SeedSet, Trajectory, TrajectoryStep,
                               build_verified_trajectory)
from value_net import ValueNetwork, ValueNetTrainer, TrainConfig


class CoverageScorer:
    """Independent finite path-coverage oracle for deterministic search tests."""
    k = 2

    def __init__(self):
        self.paths = [{0, 1}, {0, 2}, {2, 3}, {3, 4}, {4, 5}, {5}]
        self.cache = {}

    def score_many(self, sets):
        for s in sets:
            self.cache[s.nodes] = float(sum(bool(s.nodes & p) for p in self.paths))
        return [self.cache[s.nodes] for s in sets]


class MisleadingProxy:
    def predict(self, h_v, S, h_G=None):
        return -float(sum(S.nodes))

    def predict_batch(self, h_v, sets, h_G=None):
        return [self.predict(h_v, S, h_G) for S in sets]


class VerifiedSearchTests(unittest.TestCase):
    def setUp(self):
        self.hv = torch.eye(6)

    def test_each_removed_node_has_proposals_and_determinism(self):
        S = SeedSet({0, 1, 2})
        kwargs = dict(expert_nodes=[3, 4], bc_raw=torch.arange(6), seed=7)
        a = propose_mixed_neighbors(S, self.hv, **kwargs)
        b = propose_mixed_neighbors(S, self.hv, **kwargs)
        self.assertEqual(a, b)
        self.assertEqual({c.removed for c in a}, S.nodes)
        self.assertEqual(len(a), len({c.seed_set for c in a}))
        self.assertTrue(all(c.seed_set.k == 3 and c.added not in S.nodes for c in a))

    def test_exact_overrides_misleading_proxy_and_budget(self):
        scorer = CoverageScorer()
        traj = build_verified_trajectory(SeedSet({0, 1}), self.hv,
            MisleadingProxy(), scorer, H=5, per_source=5, shortlist_size=2,
            exploration_slots=1, max_batches_per_step=10, evaluation_budget=12)
        self.assertGreater(traj.terminal.score, traj.initial.score)
        self.assertTrue(all(b.score > a.score for a, b in zip(traj.steps, traj.steps[1:])))
        self.assertLessEqual(len(traj.evaluations), 12)
        self.assertEqual(len(traj.evaluations), len({s.S for s in traj.evaluations}))
        self.assertTrue(all(step.is_exact_verified for step in traj))
        self.assertTrue(traj.audit)

    def test_no_false_proxy_improvement_or_fake_optimality(self):
        class ConstantScorer(CoverageScorer):
            def score_many(self, sets):
                for s in sets:
                    self.cache[s.nodes] = 1.
                return [1.] * len(sets)
        traj = build_verified_trajectory(SeedSet({4, 5}), self.hv,
            MisleadingProxy(), ConstantScorer(), shortlist_size=2,
            exploration_slots=1, evaluation_budget=3)
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj.stop_reason, 'evaluation_budget')
        self.assertTrue(all(not r['accepted'] for r in traj.audit))

    def test_zero_budget(self):
        traj = build_verified_trajectory(SeedSet({0, 1}), self.hv,
            MisleadingProxy(), CoverageScorer(), evaluation_budget=0)
        self.assertEqual(len(traj.evaluations), 0)
        self.assertEqual(traj.stop_reason, 'evaluation_budget')

    def test_exact_label_upgrades_proxy_everywhere(self):
        S, T = SeedSet({0, 1}), SeedSet({2, 3})
        scorer = CoverageScorer()
        scorer.score_many([S])
        traj = Trajectory([TrajectoryStep(T, -50.), TrajectoryStep(S, 999.),
                           TrajectoryStep(T, -50.)])
        class Graph:
            num_nodes = 6
        rows, cache = build_dataset(Graph(), [traj], scorer)
        self.assertTrue(all(s.is_exact_verified for s in rows))
        self.assertEqual(rows[1].score, cache[S.nodes])
        self.assertEqual(rows[1].weight, 1.)
        self.assertEqual(len(canonicalize_samples(rows, cache)), 2)

    def test_archive_best_quality_diversity_and_conflicts(self):
        archive = CandidateArchive()
        for nodes, score in [({0, 1}, 10.), ({0, 2}, 9.6), ({3, 4}, 9.7), ({2, 5}, 3.)]:
            archive.add(SeedSet(nodes), score, exact=True, source='test')
        archive.add(SeedSet({0, 1}), 900., exact=False, source='proxy')
        selected = archive.select(quality_ratio=.95, max_size=3)
        self.assertEqual(selected[0].exact_score, 10.)
        self.assertEqual(selected[1].seed_set, SeedSet({3, 4}))
        self.assertTrue(all(r.exact_score >= 9.5 for r in selected))
        self.assertEqual(archive.best.exact_score, 10.)
        with self.assertRaises(ValueError):
            archive.add(SeedSet({0, 1}), 11., exact=True)

    def test_elite_weight_without_duplicate_rows(self):
        samples = [DataSample(SeedSet({0, 1}), 10., 1., True, 0, 0),
                   DataSample(SeedSet({2, 3}), 9., 1., True, 1, 0)]
        updated, _ = add_elite_buffer(samples, Phase2Config(k=2))
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0].weight, 5.)
        self.assertEqual(updated[1].weight, 3.)

    def test_crossover_exact_labels_and_rejected_observations(self):
        scorer = CoverageScorer()
        parents = [SeedSet({0, 1}), SeedSet({4, 5})]
        values = scorer.score_many(parents)
        samples = [DataSample(S, score, 1., True, i, 0, str(i))
                   for i, (S, score) in enumerate(zip(parents, values))]
        class Graph:
            num_nodes = 6
        cfg = Phase2Config(k=2, perturb_sampling=False, crossover_pairs=1,
                          crossover_quality_ratio=0., perturb_threshold=1.)
        updated, added = augment_endpoint_neighborhood(samples, Graph(), self.hv,
            scorer, cfg, reference_set=parents[0])
        self.assertEqual(len(scorer.cache), 6)  # exhaust all 2-subsets of 4-node union
        self.assertTrue(added)
        self.assertTrue(all(s.score == scorer.cache[s.seed_set.nodes] for s in added))
        self.assertTrue(all(s.score >= max(values) for s in added))
        self.assertEqual(len(updated), len({s.seed_set for s in updated}))

    def test_invalid_configuration_rejected(self):
        for kwargs in [dict(shortlist_size=0), dict(exploration_slots=8),
                       dict(candidate_max_size=0), dict(candidate_quality_ratio=1.1),
                       dict(trajectory_evaluation_budget=-1)]:
            with self.assertRaises(ValueError):
                Phase2Config(**kwargs).validate()

    def test_stable_holdout_excluded_from_every_fit(self):
        archive = CandidateArchive()
        trainer = ValueNetTrainer(ValueNetwork(6, (8,)),
                                  TrainConfig(epochs=1, log_every=0), 'cpu')
        cfg = Phase2Config(k=2, validation_ratio=.5)
        fitted = []
        original = trainer.fit
        def capture(x, y, *args, **kwargs):
            fitted.append(set(y.tolist()))
            return original(x, y, *args, **kwargs)
        trainer.fit = capture
        sets = list(itertools.combinations(range(6), 2))
        held = {}
        for batch in (sets[:8], sets):
            for i, nodes in enumerate(batch):
                archive.add(SeedSet(nodes), i+1., exact=True)
            report = fit_archive(trainer, archive, self.hv, self.hv.mean(0),
                                 cfg, epochs=1, stage='test')
            for nodes in report['validation_nodes']:
                held[tuple(nodes)] = sets.index(tuple(nodes))+1.
        self.assertTrue(held)
        self.assertTrue(all(not set(held.values()) & seen for seen in fitted))


@unittest.skipUnless(shutil.which('g++'), 'g++ required for real exact backend')
class ExactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.binary = compile_exact_gbc(Path(__file__).resolve().parents[1]/'exact_gbc.cpp',
                                       cls.root/'exact')
        # A cycle with a chord: ties and different shortest paths are present.
        cls.edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
        cls.path = cls.root/'graph.txt'
        cls.path.write_text(''.join(f'{u} {v}\n' for u, v in cls.edges))
        cls.graph, _ = load_edge_list(cls.path, directed=False)
        cls.bc = compute_node_betweenness_labels(cls.graph)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scorer(self):
        return ExactGBCScorer(self.graph, self.path, self.binary, 2, 2)

    def test_backend_matches_independent_shortest_path_enumeration(self):
        from collections import deque
        adjacency = [[] for _ in range(6)]
        for u, v in self.edges:
            adjacency[u].append(v)
            adjacency[v].append(u)
        paths = []
        for source in range(6):
            distances = [-1]*6
            distances[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v in adjacency[u]:
                    if distances[v] < 0:
                        distances[v] = distances[u]+1
                        q.append(v)
            def enumerate_paths(t):
                if t == source:
                    return [[source]]
                return [p+[t] for v in adjacency[t] if distances[v] == distances[t]-1
                        for p in enumerate_paths(v)]
            for target in range(6):
                if target != source:
                    paths.append([set(p[1:-1]) for p in enumerate_paths(target)])
        sets = [SeedSet(nodes) for nodes in itertools.combinations(range(6), 2)]
        values = self.scorer().score_many(sets)
        for S, score in zip(sets, values):
            reference = sum(sum(bool(S.nodes & p) for p in ps)/len(ps) for ps in paths)
            self.assertAlmostEqual(reference, score, places=10)

    def test_full_phase2_real_exact_and_persistence(self):
        from phase2_trajectory import run_hedge
        h_v = torch.eye(6)
        cfg = Phase2Config(k=2, H=2, bootstrap_epochs=2, retrain_epochs=2,
            online_epochs=1, online_every_trajectories=2, log_every=0,
            hidden_dims=(8,), shortlist_size=4, exploration_slots=1,
            trajectory_evaluation_budget=8, crossover_max_proposals=12,
            hedge_seeds_per_budget=1, hedge_epsilon=.5, candidate_max_size=10)
        # Reuse temporary binaries; the actual C++ HEDGE still executes.
        def hedge(*args, **kwargs):
            kwargs['binary'] = self.root/'hedge'
            return run_hedge(*args, **kwargs)
        output = self.root/'output'
        with patch('phase2_trajectory.run_hedge', side_effect=hedge):
            result = run_phase2(self.graph, h_v, self.bc, config=cfg,
                scorer=self.scorer(), centra_nodes=[1, 4], checkpoint_dir=output)
        self.assertGreaterEqual(result.best_exact.exact_score, max(result.expert_scores.values()))
        self.assertEqual(len(result.samples), len({s.seed_set for s in result.samples}))
        self.assertTrue(all(s.is_exact_verified for s in result.samples))
        self.assertTrue(all(s.score >= .95*result.best_exact.exact_score for s in result.samples))
        for traj in result.trajectory_dataset.trajectories:
            self.assertTrue(all(b.score > a.score for a, b in zip(traj.steps, traj.steps[1:])))
        payload = torch.load(output/'phase2_dataset.pt', weights_only=True)
        self.assertEqual(payload['schema_version'], 2)
        self.assertEqual(payload['best_exact']['score'], result.best_exact.exact_score)
        self.assertEqual(result.value_net_metrics[-2]['stage'], 'after_bplus')
        self.assertGreater(len(payload['candidate_archive']), 0)
        for a, b in payload['trajectory_edges']:
            self.assertEqual(len(set(payload['samples'][a]['nodes']) ^
                                 set(payload['samples'][b]['nodes'])), 2)
        summary = json.loads((output/'phase2_summary.json').read_text())
        self.assertEqual(summary['counts']['samples'], len(result.samples))

    def test_end_to_end_duplicate_and_saturated_budget(self):
        scorer = ExactGBCScorer(self.graph, self.path, self.binary, 6, 1)
        cfg = Phase2Config(k=6, H=1, bootstrap_epochs=1, retrain_epochs=1,
                          online_epochs=0, log_every=0, hidden_dims=(4,))
        result = run_phase2(self.graph, torch.eye(6), self.bc, config=cfg,
                            scorer=scorer, centra_nodes=list(range(6)))
        self.assertEqual(len(result.samples), 1)
        self.assertEqual(len(result.archive.records), 1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
