"""phase1_representation.py — Phase 1 runner for GEN-GBC.

Adapted from GEN-CIM's ``phases/phase1_representation.py`` while replacing
influence-specific supervision and structural features with GBC-aware ones.

Phase 1 pipeline
----------------
    1. Compute exact singleton betweenness labels with Brandes.
    2. Build a unified three-dimensional structural feature vector x_v:
           [ normalized balanced degree, clustering deficit, harmonic position ]
    3. Train GATv2 with pairwise ranking supervision.
    4. Save h_v and the encoder checkpoint for downstream phases.

The feature *semantics* are shared by directed and undirected graphs, while
their graph-theoretic definitions respect edge direction:

Undirected
~~~~~~~~~~
    D_v = d_v
    C_v = 2 T_v / (d_v (d_v - 1))
    H_v = 1/(|V|-1) * sum_{u != v, d(v,u)<inf} 1/d(v,u)

Directed
~~~~~~~~
    D_v = sqrt((1+d_in(v))(1+d_out(v))) - 1
    C_v = Fagiolo's binary directed clustering coefficient
    H_v = sqrt(H_in(v) H_out(v))

The resulting feature vector is always

    x_v = [ log(1+D_v)/log(1+D_max), 1-C_v, H_v ] in R^3.

Exact singleton labels and harmonic features share the same all-sources
shortest-path pass, avoiding a second O(|V||E|) traversal on unweighted
graphs.  Group scores after Phase 1 remain the responsibility of
``exact_gbc.cpp``.
"""

from __future__ import annotations

from collections import deque
import heapq
from pathlib import Path
import time
from typing import Optional, Union

import torch
from torch import Tensor

from gbc_types import GraphData
from gatv2 import Phase1Config, Phase1Trainer


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Exact singleton GBC labels + harmonic shortest-path statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _build_adjacency(graph: GraphData) -> list[list[tuple[int, float]]]:
    """Return outgoing adjacency with unit or positive path costs."""
    edges = graph.edge_index.detach().cpu().t().tolist()
    costs = (
        graph.edge_weight.detach().cpu().tolist()
        if graph.edge_weight is not None
        else None
    )
    adjacency: list[list[tuple[int, float]]] = [
        [] for _ in range(graph.num_nodes)
    ]
    for i, (u, v) in enumerate(edges):
        if u != v:
            adjacency[u].append(
                (v, float(costs[i]) if costs is not None else 1.0)
            )
    return adjacency


def _compute_brandes_statistics(
    graph: GraphData,
    *,
    verbose: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute singleton BC and in/out harmonic centrality in one pass.

    The Brandes dependency accumulation yields exact raw singleton GBC labels.
    During the same all-sources shortest-path traversal we accumulate

        H_out(v) = 1/(N-1) * sum_u 1/d(v,u)
        H_in (v) = 1/(N-1) * sum_u 1/d(u,v)

    over reachable u != v.  For undirected graphs, the loader stores both arc
    orientations and therefore H_in == H_out (up to floating-point roundoff).

    Returns
    -------
    labels : Tensor [N], CPU float64
        Exact, unnormalized ordered-pair internal-node singleton GBC.
    harmonic_in : Tensor [N], CPU float64
        Normalized incoming harmonic centrality.
    harmonic_out : Tensor [N], CPU float64
        Normalized outgoing harmonic centrality.
    """
    n = graph.num_nodes
    adjacency = _build_adjacency(graph)
    weighted = graph.edge_weight is not None
    labels = [0.0] * n
    harmonic_in = [0.0] * n
    harmonic_out = [0.0] * n

    started = time.perf_counter()
    report_every = max(1, n // 10)
    if verbose:
        print(
            f"[Phase1/Brandes] Đang tính exact singleton GBC + harmonic "
            f"statistics cho {n} node từ {graph.num_edges} cung; "
            f"{'Dijkstra' if weighted else 'BFS'}; "
            f"directed={graph.directed}.",
            flush=True,
        )

    for source in range(n):
        predecessors: list[list[int]] = [[] for _ in range(n)]
        sigma = [0.0] * n
        sigma[source] = 1.0
        stack: list[int] = []

        if not weighted:
            # Unweighted single-source shortest paths.
            distance = [-1] * n
            distance[source] = 0
            queue = deque([source])
            while queue:
                v = queue.popleft()
                stack.append(v)
                for w, _ in adjacency[v]:
                    if distance[w] < 0:
                        distance[w] = distance[v] + 1
                        queue.append(w)
                    if distance[w] == distance[v] + 1:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            for target, dist in enumerate(distance):
                if target != source and dist > 0:
                    reciprocal = 1.0 / dist
                    harmonic_out[source] += reciprocal
                    harmonic_in[target] += reciprocal

        else:
            # Positive costs guarantee that shortest-path predecessors have
            # strictly smaller distances than their targets.
            distance = [float("inf")] * n
            distance[source] = 0.0
            heap = [(0.0, source)]
            while heap:
                distance_v, v = heapq.heappop(heap)
                if distance_v != distance[v]:
                    continue
                stack.append(v)
                for w, cost in adjacency[v]:
                    candidate = distance_v + cost
                    if candidate < distance[w]:
                        distance[w] = candidate
                        heapq.heappush(heap, (candidate, w))
                        sigma[w] = sigma[v]
                        predecessors[w] = [v]
                    elif candidate == distance[w]:
                        sigma[w] += sigma[v]
                        predecessors[w].append(v)

            for target, dist in enumerate(distance):
                if target != source and dist != float("inf"):
                    reciprocal = 1.0 / dist
                    harmonic_out[source] += reciprocal
                    harmonic_in[target] += reciprocal

        # Standard Brandes reverse dependency accumulation.  Because an
        # undirected edge is stored by two COO arcs and every source is run
        # once, undirected labels count ordered source-target pairs; do not
        # divide by two.
        dependency = [0.0] * n
        while stack:
            w = stack.pop()
            if sigma[w]:
                scale = (1.0 + dependency[w]) / sigma[w]
                for v in predecessors[w]:
                    dependency[v] += sigma[v] * scale
            if w != source:
                labels[w] += dependency[w]

        completed = source + 1
        if verbose and (completed % report_every == 0 or completed == n):
            elapsed = time.perf_counter() - started
            print(
                f"[Phase1/Brandes] {completed}/{n} node "
                f"({100 * completed / n:.1f}%), đã chạy {elapsed:.1f}s.",
                flush=True,
            )

    normalizer = float(max(1, n - 1))
    return (
        torch.tensor(labels, dtype=torch.float64),
        torch.tensor(harmonic_in, dtype=torch.float64) / normalizer,
        torch.tensor(harmonic_out, dtype=torch.float64) / normalizer,
    )


def compute_node_betweenness_labels(graph: GraphData) -> Tensor:
    r"""Compute exact raw GBC({v}) for every v using Brandes accumulation.

    The score convention matches ``exact_gbc.cpp``: reachable ordered
    source-target pairs contribute the fraction of their shortest paths whose
    *internal* nodes contain v.  Endpoints do not receive credit.

    Parameters
    ----------
    graph : GraphData
        Directed COO arcs, or both orientations for every undirected edge.

    Returns
    -------
    Tensor [N], CPU float64
        Exact, unnormalized singleton GBC labels.
    """
    labels, _, _ = _compute_brandes_statistics(graph)
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  2. GBC-aware structural node features
# ═══════════════════════════════════════════════════════════════════════════════

def _balanced_degree(graph: GraphData) -> Tensor:
    r"""Return D_v used by the first GBC structural feature.

    Undirected:
        D_v = d_v.

    Directed:
        D_v = sqrt((1+d_in(v))(1+d_out(v))) - 1.

    The directed definition rewards nodes that can participate on both sides
    of an internal directed path while remaining finite for sources/sinks.
    """
    src = graph.edge_index[0].detach().cpu()
    dst = graph.edge_index[1].detach().cpu()
    out_degree = torch.bincount(src, minlength=graph.num_nodes).to(torch.float64)

    if not graph.directed:
        return out_degree

    in_degree = torch.bincount(dst, minlength=graph.num_nodes).to(torch.float64)
    return torch.sqrt((1.0 + in_degree) * (1.0 + out_degree)) - 1.0


def _undirected_clustering(graph: GraphData) -> Tensor:
    r"""Compute C_v = 2T_v / (d_v(d_v-1)) for an undirected graph.

    The GEN-GBC loader represents every undirected edge by both COO
    orientations.  Neighbor sets therefore contain the ordinary undirected
    neighborhood exactly once after set deduplication.
    """
    n = graph.num_nodes
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for u, v in graph.edge_index.detach().cpu().t().tolist():
        if u != v:
            neighbors[u].add(v)

    clustering = torch.zeros(n, dtype=torch.float64)
    for v in range(n):
        neighborhood = neighbors[v]
        degree = len(neighborhood)
        if degree < 2:
            continue

        # For every neighbor u, count common neighbors of u and v.  Each
        # triangle through v is counted twice, exactly matching 2*T_v.
        twice_triangles = sum(
            len(neighborhood.intersection(neighbors[u]))
            for u in neighborhood
        )
        clustering[v] = twice_triangles / (degree * (degree - 1))

    return clustering.clamp_(0.0, 1.0)


def _directed_clustering(graph: GraphData) -> Tensor:
    r"""Compute Fagiolo's binary directed clustering coefficient.

    Let A be the binary directed adjacency matrix, k_tot = k_in + k_out, and
    k_rec the number of reciprocated neighbors.  Then

        C_v = [(A + A^T)^3]_{vv}
              / {2 [k_tot(v)(k_tot(v)-1) - 2 k_rec(v)]}.

    The sparse set/dictionary implementation below evaluates the same formula
    without materializing an N x N dense adjacency matrix.
    """
    n = graph.num_nodes
    out_neighbors: list[set[int]] = [set() for _ in range(n)]
    in_neighbors: list[set[int]] = [set() for _ in range(n)]
    for u, v in graph.edge_index.detach().cpu().t().tolist():
        if u != v:
            out_neighbors[u].add(v)
            in_neighbors[v].add(u)

    # B = A + A^T is symmetric.  B_vu is 1 for a one-way dyad and 2 for a
    # reciprocated dyad.
    symmetric_weight: list[dict[int, int]] = []
    for v in range(n):
        local: dict[int, int] = {}
        for u in out_neighbors[v] | in_neighbors[v]:
            local[u] = int(u in out_neighbors[v]) + int(u in in_neighbors[v])
        symmetric_weight.append(local)

    clustering = torch.zeros(n, dtype=torch.float64)
    for v in range(n):
        k_in = len(in_neighbors[v])
        k_out = len(out_neighbors[v])
        k_total = k_in + k_out
        k_recip = len(in_neighbors[v].intersection(out_neighbors[v]))
        denominator = 2.0 * (
            k_total * (k_total - 1) - 2 * k_recip
        )
        if denominator <= 0:
            continue

        # Diagonal entry [(A + A^T)^3]_{vv}.
        numerator = 0.0
        weights_v = symmetric_weight[v]
        for j, weight_vj in weights_v.items():
            weights_j = symmetric_weight[j]
            # Iterate over the smaller dictionary for a sparse intersection.
            if len(weights_j) <= len(weights_v):
                iterator = weights_j.items()
                for k, weight_jk in iterator:
                    weight_vk = weights_v.get(k)
                    if weight_vk is not None:
                        numerator += weight_vj * weight_jk * weight_vk
            else:
                for k, weight_vk in weights_v.items():
                    weight_jk = weights_j.get(k)
                    if weight_jk is not None:
                        numerator += weight_vj * weight_jk * weight_vk

        clustering[v] = numerator / denominator

    return clustering.clamp_(0.0, 1.0)


def build_node_features(
    graph: GraphData,
    harmonic_in: Optional[Tensor] = None,
    harmonic_out: Optional[Tensor] = None,
) -> Tensor:
    r"""Build the unified three-dimensional GBC structural feature matrix.

    For every node v,

        x_v = [
            log(1 + D_v) / log(1 + D_max),
            1 - C_v,
            H_v,
        ].

    Undirected graph
    ----------------
        D_v = ordinary degree
        C_v = 2 T_v / (d_v (d_v - 1)), with C_v = 0 when d_v < 2
        H_v = normalized harmonic centrality

    Directed graph
    --------------
        D_v = sqrt((1+d_in)(1+d_out)) - 1
        C_v = Fagiolo binary directed clustering coefficient
        H_v = sqrt(H_in(v) H_out(v))

    ``C_v`` is topological (edge weights are ignored), while shortest-path
    costs are respected by ``H_v`` whenever ``graph.edge_weight`` is present.

    Parameters
    ----------
    graph : GraphData
    harmonic_in, harmonic_out : Tensor [N] | None
        Optional precomputed normalized harmonic statistics.  Supplying both
        lets ``run_phase1`` reuse the Brandes traversal used for exact labels.

    Returns
    -------
    Tensor [N, 3] on graph.device, dtype float32
    """
    if (harmonic_in is None) != (harmonic_out is None):
        raise ValueError("harmonic_in and harmonic_out must be supplied together")

    if harmonic_in is None:
        # Standalone calls remain convenient; run_phase1 avoids this extra
        # traversal by supplying the statistics from _compute_brandes_statistics.
        _, harmonic_in, harmonic_out = _compute_brandes_statistics(
            graph, verbose=False
        )

    assert harmonic_in is not None and harmonic_out is not None
    harmonic_in = harmonic_in.detach().cpu().to(torch.float64)
    harmonic_out = harmonic_out.detach().cpu().to(torch.float64)
    if harmonic_in.shape != (graph.num_nodes,) or harmonic_out.shape != (
        graph.num_nodes,
    ):
        raise ValueError("harmonic statistics must have shape [num_nodes]")
    if not bool(torch.isfinite(harmonic_in).all()) or not bool(
        torch.isfinite(harmonic_out).all()
    ):
        raise ValueError("harmonic statistics must be finite")

    degree = _balanced_degree(graph)
    maximum = degree.max() if degree.numel() else torch.tensor(0.0)
    log_denominator = torch.log1p(maximum)
    if float(log_denominator) > 0.0:
        feature_degree = torch.log1p(degree) / log_denominator
    else:
        feature_degree = torch.zeros_like(degree)

    clustering = (
        _directed_clustering(graph)
        if graph.directed
        else _undirected_clustering(graph)
    )
    feature_bridge = 1.0 - clustering

    if graph.directed:
        harmonic = torch.sqrt(
            torch.clamp_min(harmonic_in, 0.0)
            * torch.clamp_min(harmonic_out, 0.0)
        )
    else:
        # For a correctly stored undirected graph H_in == H_out.  Use their
        # mean to suppress tiny numerical differences on weighted inputs.
        harmonic = 0.5 * (harmonic_in + harmonic_out)

    features = torch.stack(
        (feature_degree, feature_bridge, harmonic), dim=1
    ).to(torch.float32)
    return features.to(graph.device)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Main Phase 1 runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(
    graph: GraphData,
    node_features: Optional[Tensor] = None,
    config: Optional[Phase1Config] = None,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_dir: Union[str, Path] = "experiments/checkpoints",
    dataset_name: str = "graph",
    return_labels: bool = False,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """Run exact labels -> GBC features -> GATv2 ranking -> node embeddings.

    Parameters
    ----------
    graph : GraphData
        Directed arcs, or both orientations for every undirected edge.
        ``graph.directed`` selects the directed/undirected feature definitions.
    node_features : Tensor [N, d_in] | None
        If None, build the three GBC-aware structural features.  User-supplied
        raw features remain supported and bypass structural feature creation.
    config : Phase1Config | None
        GATv2 and optimizer configuration.
    device : torch.device | str | None
        Default: CUDA if available, otherwise CPU.
    checkpoint_dir : str | Path
        Where to save the encoder checkpoint and h_v tensor.
    dataset_name : str
        Suffix used in output file names.
    return_labels : bool
        When True, also return exact singleton labels for Phase 2 reuse.

    Returns
    -------
    Tensor [N, hidden_channels] | (Tensor [N, hidden_channels], Tensor [N])
        Learned node embeddings, optionally with exact singleton labels.
    """
    if graph.num_nodes < 1:
        raise ValueError("Graph is empty")

    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"[Phase1] Graph type: {'directed' if graph.directed else 'undirected'}; "
        f"N={graph.num_nodes}, COO arcs={graph.num_edges}.",
        flush=True,
    )

    start = time.perf_counter()
    if node_features is None:
        print(
            "[Phase1] Không có raw node features — dùng 3 GBC-aware features: "
            "normalized balanced degree, 1-C_v, harmonic position.",
            flush=True,
        )
        labels, harmonic_in, harmonic_out = _compute_brandes_statistics(graph)
        node_features = build_node_features(
            graph, harmonic_in=harmonic_in, harmonic_out=harmonic_out
        )
    else:
        print(
            "[Phase1] Dùng node_features do caller cung cấp; "
            "vẫn tạo exact singleton GBC labels bằng Brandes.",
            flush=True,
        )
        labels = compute_node_betweenness_labels(graph)

    if (
        node_features.ndim != 2
        or node_features.size(0) != graph.num_nodes
        or node_features.size(1) < 1
    ):
        raise ValueError(
            "node_features must have shape [num_nodes, feature_dim]"
        )
    if not bool(torch.isfinite(node_features).all()):
        raise ValueError("node_features must be finite")

    elapsed = time.perf_counter() - start
    print(
        f"[Phase1] Chuẩn bị features + labels xong sau {elapsed:.1f}s; "
        f"x={tuple(node_features.shape)}, "
        f"GBC min={labels.min().item():.3f}, "
        f"max={labels.max().item():.3f}, "
        f"mean={labels.mean().item():.3f}.",
        flush=True,
    )
    if node_features.size(1) == 3:
        mins = node_features.detach().cpu().min(dim=0).values.tolist()
        maxs = node_features.detach().cpu().max(dim=0).values.tolist()
        print(
            "[Phase1] Feature ranges "
            f"D=[{mins[0]:.4f},{maxs[0]:.4f}], "
            f"1-C=[{mins[1]:.4f},{maxs[1]:.4f}], "
            f"H=[{mins[2]:.4f},{maxs[2]:.4f}].",
            flush=True,
        )

    trainer = Phase1Trainer(
        node_features.size(1), config=config, device=dev
    )
    print(f"[Phase1] Training GATv2 on {dev}", flush=True)
    train_start = time.perf_counter()
    losses = trainer.fit(node_features, graph.edge_index, labels)
    train_elapsed = time.perf_counter() - train_start

    h_v = trainer.encode(node_features, graph.edge_index)

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trainer.save(directory / f"phase1_{dataset_name}.pt")
    torch.save(
        h_v.detach().cpu(), directory / f"h_v_{dataset_name}.pt"
    )

    print(
        f"[Phase1] h_v={tuple(h_v.shape)}, epochs={len(losses)}, "
        f"best_loss={min(losses):.6f}, train={train_elapsed:.1f}s",
        flush=True,
    )
    return (h_v, labels) if return_labels else h_v


# ═══════════════════════════════════════════════════════════════════════════════
#  Smoke tests
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test() -> None:
    """Check labels, both feature definitions, training, and saved files."""
    print("=" * 72)
    print("  phase1_representation.py — GEN-GBC Smoke Tests")
    print("=" * 72)
    torch.manual_seed(0)

    # ── Exact singleton labels ─────────────────────────────────────────────
    # Undirected path 0--1--2: node 1 lies internally on ordered pairs
    # (0,2) and (2,0).
    path_u = GraphData(
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        3,
    )
    assert compute_node_betweenness_labels(path_u).tolist() == [0.0, 2.0, 0.0]
    print("✓ Test 1  undirected ordered-pair singleton GBC: [0, 2, 0]")

    path_d = GraphData(
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        3,
        directed=True,
    )
    assert compute_node_betweenness_labels(path_d).tolist() == [0.0, 1.0, 0.0]
    print("✓ Test 2  directed singleton GBC: [0, 1, 0]")

    # Two equal shortest paths 0->1->3 and 0->2->3 split one unit of credit.
    diamond = GraphData(
        torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        4,
        directed=True,
    )
    assert compute_node_betweenness_labels(diamond).tolist() == [
        0.0, 0.5, 0.5, 0.0
    ]
    print("✓ Test 3  multiple shortest paths split fractional credit")

    weighted = GraphData(
        torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        3,
        edge_weight=torch.tensor([1.0, 3.0, 1.0]),
        directed=True,
    )
    assert compute_node_betweenness_labels(weighted).tolist() == [0.0, 1.0, 0.0]
    print("✓ Test 4  weighted Dijkstra follows the cheaper two-hop path")

    # ── Undirected features ────────────────────────────────────────────────
    # Triangle: normalized degree=1, C=1 -> 1-C=0, harmonic=1.
    triangle_u = GraphData(
        torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
            dtype=torch.long,
        ),
        3,
    )
    triangle_features = build_node_features(triangle_u)
    assert torch.allclose(
        triangle_features,
        torch.tensor([[1.0, 0.0, 1.0]] * 3),
        atol=1e-6,
    )
    print("✓ Test 5  undirected features: degree / clustering / harmonic")

    path_features = build_node_features(path_u)
    expected_degree = torch.tensor([
        torch.log(torch.tensor(2.0)) / torch.log(torch.tensor(3.0)),
        1.0,
        torch.log(torch.tensor(2.0)) / torch.log(torch.tensor(3.0)),
    ])
    assert torch.allclose(path_features[:, 0], expected_degree, atol=1e-6)
    assert torch.allclose(path_features[:, 1], torch.ones(3), atol=1e-6)
    assert torch.allclose(
        path_features[:, 2], torch.tensor([0.75, 1.0, 0.75]), atol=1e-6
    )
    print("✓ Test 6  undirected path feature values are exact")

    # ── Directed features ──────────────────────────────────────────────────
    # Directed chain 0->1->2:
    #   balanced D = [sqrt(2)-1, 1, sqrt(2)-1] -> normalized [0.5,1,0.5]
    #   no directed triangle -> 1-C = 1
    #   H = sqrt(H_in H_out) = [0, 0.5, 0]
    directed_features = build_node_features(path_d)
    assert torch.allclose(
        directed_features[:, 0], torch.tensor([0.5, 1.0, 0.5]), atol=1e-6
    )
    assert torch.allclose(
        directed_features[:, 1], torch.ones(3), atol=1e-6
    )
    assert torch.allclose(
        directed_features[:, 2], torch.tensor([0.0, 0.5, 0.0]), atol=1e-6
    )
    print("✓ Test 7  directed balanced-degree and in/out harmonic semantics")

    # A fully reciprocal directed triangle is the directed representation of
    # an undirected triangle; Fagiolo C_v must reduce to 1.
    triangle_d = GraphData(
        triangle_u.edge_index.clone(), 3, directed=True
    )
    directed_triangle_features = build_node_features(triangle_d)
    assert torch.allclose(
        directed_triangle_features[:, 1], torch.zeros(3), atol=1e-6
    )
    print("✓ Test 8  Fagiolo clustering reduces correctly on reciprocal triangle")

    if not torch_geometric_available():
        print("[SKIP] Tests 9–10 require torch_geometric")
        return

    from tempfile import TemporaryDirectory

    config = Phase1Config(
        hidden_channels=16,
        heads=4,
        n_epochs=3,
        patience=5,
        log_every=0,
        n_pairs=32,
    )
    with TemporaryDirectory() as temp:
        h_v = run_phase1(
            path_u,
            config=config,
            device="cpu",
            checkpoint_dir=temp,
            dataset_name="undirected",
        )
        assert h_v.shape == (3, 16)
        assert (Path(temp) / "phase1_undirected.pt").is_file()
        assert torch.load(
            Path(temp) / "h_v_undirected.pt",
            weights_only=True,
        ).shape == (3, 16)
        print("✓ Test 9  undirected end-to-end training and saved embeddings")

        directed_hv = run_phase1(
            path_d,
            config=config,
            device="cpu",
            checkpoint_dir=temp,
            dataset_name="directed",
        )
        assert directed_hv.shape == (3, 16)
        assert (Path(temp) / "phase1_directed.pt").is_file()
        print("✓ Test 10 directed end-to-end training and checkpoint")

    print("  All 10 phase1_representation.py smoke tests passed ✓")


def torch_geometric_available() -> bool:
    """Return whether the optional PyG dependency required by GATv2 exists."""
    from gatv2 import GATv2Conv
    return GATv2Conv is not None


if __name__ == "__main__":
    _smoke_test()
