# GEN-GBC

GEN-CIM-inspired group betweenness maximization. Phase 1 remains GATv2
trained on exact singleton BC; the current update implements verified Phase 2
search and a unique, quality/diversity-selected candidate dataset.

## Install and check

Python 3.10+, PyTorch and a C++17 compiler with OpenMP are required for Phase 2.
Phase 1 training additionally requires `torch-geometric`.

```bash
pip install torch
pip install torch-geometric  # only for training Phase 1
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Use the existing CPU PyTorch in Colab. No GPU is required. Importable entry
points `dataset_builder.py`, `phase2_trajectory.py`, and `value_net.py` load
their numbered source files. Keep both when cloning. Existing Colab cells
that download a numbered source to its standard module name still work;
also download the new `candidate_archive.py` and updated `neighbor.py`.

## Run from Phase 1 outputs

```python
from graph_utils import load_edge_list
from dataset_builder import Phase2Config, run_phase2
import torch

graph, _ = load_edge_list('ca-GrQc.txt', directed=False)
h_v = torch.load('h_v_ca-grqc.pt', map_location='cpu', weights_only=True)
bc_raw = torch.load('bc_raw.pt', map_location='cpu', weights_only=True)

cfg = Phase2Config(k=10, seed=42, exact_threads=2)
result = run_phase2(graph, h_v, bc_raw, config=cfg,
                    graph_path='ca-GrQc.txt', device='cpu',
                    checkpoint_dir='experiments/ca-grqc-k10')

print('Best exact:', result.best_exact.exact_score)
print('Best nodes (internal IDs):', sorted(result.best_exact.seed_set.nodes))
print('Unique Dcand:', len(result.samples))
for sample in result.samples:
    print(sorted(sample.seed_set.nodes), sample.score, sample.weight)
```

Embeddings, BC labels and the edge list **must use the same sorted internal
ID mapping**. Existing singleton checks remain enabled. Scores count ordered
source/target pairs with at least one group node strictly inside the shortest
path; they are not NetworkX's group score excluding group endpoints.

CLI:

```bash
python dataset_builder.py --graph ca-GrQc.txt --embeddings h_v_ca-grqc.pt \
  --bc-raw bc_raw.pt --k 10 --threads 2 --device cpu \
  --trajectory-evaluation-budget 64 --shortlist-size 8 --exploration-slots 2 \
  --candidate-quality-ratio 0.95 --candidate-max-size 256 \
  --output experiments/ca-grqc-k10
```

## What changed

1. Each removed seed has cosine, expert, singleton-BC and random proposals.
   The default quota is two per deterministic source and one random proposal,
   deduplicated before scoring. There is no global cosine-only proposal cap.
2. ValueNet ranks proposals. Exact GBC accepts a move only when
   `new > old + improvement_atol + improvement_rtol * abs(old)`.
3. Each batch has up to 8 candidates, including up to 2 exploratory slots.
   At most two batches are tried per step and 64 distinct candidates per
   trajectory. Budget counts cache hits too; the initial expert is excluded.
   `H=5` remains the step limit. A failed shortlist is not a proof of local
   optimality; stop reasons are recorded.
4. All paid-for exact labels, including unselected HEDGE trials and rejected
   moves/B+ proposals, are kept for ValueNet. Updates occur every 3
   trajectories (50 epochs) and after trajectories and B+ (500 epochs each
   by default). Target calibration remains frozen after bootstrap.
5. A deterministic hash partitions whole sets into training/validation;
   one stable training anchor ensures bootstrap has data. Held-out sets are
   never fitted, including during online updates. Because exact search is
   adaptive, this is a diagnostic, **not an independent benchmark test**.
   Prequential logs also compare predictions with exact results before those
   new labels are fitted. No model selection uses validation here.
6. Dcand contains only unique exact sets scoring at least 95% of the best
   observed score, selected by farthest-first node diversity up to 256 sets.
   The best is always retained. Minimum node replacement distance defaults
   to 1. Sparse archives are not padded. These thresholds are configurable
   experimental choices, not guarantees from GEN-CIM.
7. Elite weighting boosts an existing row (base confidence + elite boost),
   without appending duplicate training rows. The full exact archive keeps
   the incumbent independently of Dcand selection.

Changing `verified_search=False` (CLI `--no-verified-search`) selects the
legacy proxy-only trajectory builder, **not a reproduction of the entire
old pipeline**: archive, final data selection and training fixes remain on.
Use the parent Git commit for a faithful old-versus-new baseline.

## Saved interface for Phase 3/4

`phase2_dataset.pt` is a plain-dictionary **schema version 2** payload:

- `samples`: one exact row per selected set; confidence and elite boost are
  represented in `weight`. `trajectory_idx` is a unique singleton ID, not
  an inferred trajectory. Source labels are also retained.
- `trajectory_edges`: index pairs into `samples` for original consecutive
  trajectory steps **only when both survive selection**. Use these edges
  for trajectory smoothness; do not infer edges by grouping canonical rows.
- `trajectories`: full steps, exact verification flags, stop reasons and
  prequential per-move audit, independently of Dcand filtering.
- `candidate_archive`: all canonical observed labels and source provenance.
  An exact label always overrides a proxy for the same set.
- `best_exact`, `exact_cache`, `config`, `graph_fingerprint`, timings and
  `value_net_metrics` support auditing and later integration.

`phase2_summary.json` contains human-readable counts, candidates and metrics.
`phase2_value_net.pt` is retrained **after B+**. Phase 3/4 are not implemented
in this repository yet; downstream smoothness consumers must adopt the
explicit edge interface above. This update does not add path-coverage repair
or overlap supervision to Phase 1.

## Validation and experimental limits

The regression suite covers misleading proxies, strict exact improvement,
budgets, unique labels/elite rows, stable holdout, independent shortest-path
enumeration, and end-to-end Phase 2 with real C++ exact/HEDGE backends on a
small graph. It also checks saturated budgets and saved artifacts.

These checks establish implementation behavior; they do not establish a
gain over CentRA on SNAP graphs. Compare the parent baseline and this version
on ca-grqc k=10 first, then p2p-gnutella08 and lastfm-asia, using multiple seeds
and recording exact-call counts and wall time. Include an equal-budget
expert-plus-local-search comparator before attributing gains to generative
learning. More exact verification can increase CPU time.
