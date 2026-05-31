"""
Pathfinding benchmark.

Compares the Connectome Interpreter pathfinding against the general-purpose
graph libraries commonly used in neuroscience (networkx, networkit).

Two outputs of our method are timed SEPARATELY, because they are different
objects with very different costs:

  1. subgraph extraction  -- `find_paths_of_length` returns the edge-induced
     subgraph (the connection set) relating all source/target neurons on
     pathways of the given length. This is the primary output used in most
     analyses and avoids the combinatorial blow-up of listing every path.

  2. full enumeration      -- `enumerate_paths` takes the extracted subgraph
     and lists every individual path. This is the object the baselines
     (networkx all_simple_paths, networkit AllSimplePaths) compute, so it is
     the apples-to-apples comparison for path *enumeration*.

For each query we record:
  - t_subgraph  : time to extract the subgraph (our method, cumulative over
                  path lengths 2..L)
  - t_enumerate : ADDITIONAL time to enumerate all paths from the subgraph
  - t_total     : t_subgraph + t_enumerate
  - n_connections : number of edges in the extracted subgraph
  - n_paths     : number of enumerated paths
"""

import random
import time

import numpy as np
import pandas as pd
import scipy as sp
import networkx as nx
import networkit as nk
from tqdm import tqdm

from connectome_interpreter import find_paths_of_length, enumerate_paths

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
inprop = sp.sparse.load_npz(
    "../data/adult_inprop_cb_neuron_no_CX_axonic_postsynapses.npz"
)
meta = pd.read_csv(
    "../data/adult_cb_neuron_meta_no_CX_axonic_postsynapses.csv"
)

# Cell types represented by a single neuron are excluded so that source/target
# groups contain multiple cells (a more representative workload).
sctypes = meta.groupby("cell_type").root_id.nunique()
sctypes = sctypes.index[sctypes == 1]

sensory_types = meta[
    (meta.super_class == "sensory") & (~meta.cell_type.isin(sctypes))
].cell_type.unique()
dn_types = meta[
    (meta.super_class == "descending") & (~meta.cell_type.isin(sctypes))
].cell_type.unique()

random.seed(42)
selected_sensory_types = random.sample(list(sensory_types), 5)
selected_dn_types = random.sample(list(dn_types), 5)

PATH_LENGTHS = [2, 3, 4, 5]
# networkx / networkit enumeration becomes intractable at length 5 on this
# graph; cap their workload to avoid runs that never terminate.
BASELINE_MAX_LENGTH = 4


# --------------------------------------------------------------------------- #
# Connectome Interpreter: subgraph extraction + enumeration, timed separately
# --------------------------------------------------------------------------- #
def benchmark_ours():
    results = []
    for length in tqdm(PATH_LENGTHS, desc="ours"):
        for source_type in selected_sensory_types:
            for dn_type in selected_dn_types:
                inidx = meta.idx[meta.cell_type == source_type]
                outidx = meta.idx[meta.cell_type == dn_type]

                # --- 1. subgraph extraction (cumulative over lengths 2..L) ---
                t0 = time.time()
                subgraph = None
                for plength in range(2, length + 1):
                    subgraph = find_paths_of_length(
                        inprop, inidx, outidx, plength, quiet=True
                    )
                t_subgraph = time.time() - t0

                n_connections = 0 if subgraph is None else len(subgraph)

                # --- 2. full enumeration from the extracted subgraph ---------
                # Only enumerate if a subgraph was found.
                if subgraph is not None and len(subgraph) > 0:
                    t1 = time.time()
                    n_paths = sum(1 for _ in enumerate_paths(
                        subgraph, start_layer=1, end_layer=length, return_generator=True
                    ))
                    t_enumerate = time.time() - t1
                else:
                    t_enumerate = 0.0
                    n_paths = 0

                results.append(
                    {
                        "method": "connectome_interpreter",
                        "source_type": source_type,
                        "dn_type": dn_type,
                        "path_length": length,
                        "t_subgraph": t_subgraph,
                        "t_enumerate": t_enumerate,
                        "t_total": t_subgraph + t_enumerate,
                        "source_count": len(inidx),
                        "target_count": len(outidx),
                        "n_connections": n_connections,
                        "n_paths": n_paths,
                    }
                )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# networkx: all_simple_paths enumeration
# --------------------------------------------------------------------------- #
def benchmark_networkx(G):
    results = []
    for length in tqdm(
        [l for l in PATH_LENGTHS if l <= BASELINE_MAX_LENGTH], desc="networkx"
    ):
        for source_type in selected_sensory_types:
            for dn_type in selected_dn_types:
                sources = meta.idx[meta.cell_type == source_type].values
                targets = set(meta.idx[meta.cell_type == dn_type].values)

                t0 = time.time()
                n_paths = 0
                for s in sources:
                    try:
                        for path in nx.all_simple_paths(
                            G, source=s, target=targets, cutoff=length
                        ):
                            # exact length only (edges = nodes - 1)
                            if len(path) - 1 == length:
                                n_paths += 1
                    except nx.NetworkXNoPath:
                        pass
                t_total = time.time() - t0

                results.append(
                    {
                        "method": "networkx",
                        "source_type": source_type,
                        "dn_type": dn_type,
                        "path_length": length,
                        "t_total": t_total,
                        "source_count": len(sources),
                        "target_count": len(targets),
                        "n_paths": n_paths,
                    }
                )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# networkit: AllSimplePaths enumeration
# --------------------------------------------------------------------------- #
def nk_count_paths_fixed_len(G, sources, targets, L):
    total = 0
    for s in sources:
        for t in targets:
            if s == t:
                continue
            asp = nk.reachability.AllSimplePaths(G, int(s), int(t), cutoff=L)
            try:
                asp.run()
            except RuntimeError as e:
                if "cannot reach target" in str(e):
                    continue
                raise
            cnt = 0

            def cb(path):
                nonlocal cnt
                if len(path) - 1 == L:
                    cnt += 1

            asp.forAllSimplePaths(cb)
            total += cnt
    return total


def benchmark_networkit(G):
    results = []
    for length in tqdm(
        [l for l in PATH_LENGTHS if l <= BASELINE_MAX_LENGTH], desc="networkit"
    ):
        for source_type in selected_sensory_types:
            for dn_type in selected_dn_types:
                sources = meta.idx[meta.cell_type == source_type].astype(int).values
                targets = meta.idx[meta.cell_type == dn_type].astype(int).values

                t0 = time.time()
                n_paths = nk_count_paths_fixed_len(G, sources, targets, length)
                t_total = time.time() - t0

                results.append(
                    {
                        "method": "networkit",
                        "source_type": source_type,
                        "dn_type": dn_type,
                        "path_length": length,
                        "t_total": t_total,
                        "source_count": len(sources),
                        "target_count": len(targets),
                        "n_paths": n_paths,
                    }
                )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Our method
    ours_df = benchmark_ours()
    ours_df.to_csv("pathfinding_ours_benchmark.csv", index=False)

    # networkx
    G_nx = nx.from_scipy_sparse_array(inprop, create_using=nx.DiGraph)
    nx_df = benchmark_networkx(G_nx)
    nx_df.to_csv("pathfinding_networkx_benchmark.csv", index=False)

    # networkit
    G_nk = nk.Graph(inprop.shape[0], weighted=True, directed=True)
    coo = inprop.tocoo()
    for u, v, w in zip(coo.row, coo.col, coo.data):
        G_nk.addEdge(int(u), int(v), float(w))
    G_nk = nk.graphtools.toUnweighted(G_nk)
    nk_df = benchmark_networkit(G_nk)
    nk_df.to_csv("pathfinding_networkit_benchmark.csv", index=False)

    print("Done. Wrote:")
    print("  pathfinding_ours_benchmark.csv")
    print("  pathfinding_networkx_benchmark.csv")
    print("  pathfinding_networkit_benchmark.csv")