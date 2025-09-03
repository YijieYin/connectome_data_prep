import numpy as np
import pandas as pd
import torch
from connectome_interpreter import *
from nglscenes import *
import scipy as sp
from tqdm import tqdm
import plotly.express as px
from scipy.cluster.hierarchy import linkage, leaves_list
import networkit as nk

from typing import Iterable, Union, Optional, Callable

inprop = sp.sparse.load_npz("../data/adult_inprop_cb_neuron_no_CX_axonic_postsynapses.npz")
meta = pd.read_csv("../data/adult_cb_neuron_meta_no_CX_axonic_postsynapses.csv")

# steps = compress_paths(inprop, 5)

# find_paths_within_length
import networkx as nx
import time
from itertools import product

def find_paths_within_length(G, source_indices, target_indices, path_length):
    """
    Find all paths of specified length between source and target indices in a graph.

    Parameters:
    -----------
    G : networkx.Graph
        The graph to search
    source_indices : list
        List of source node indices
    target_indices : list
        List of target node indices
    path_length : int
        The desired path length

    Returns:
    --------
    dict
        Dictionary with (source, target) tuples as keys and lists of paths as values
    """
    # Start timing
    start_time = time.time()

    # Dictionary to store the results
    paths_dict = {}

    # Convert target_indices to a set for faster lookup
    target_set = set(target_indices)

    # Profiling variables
    source_count = 0
    total_paths_found = 0

    for source in source_indices:
        source_count += 1
        source_start_time = time.time()
        source_paths_count = 0

        try:
            # Get all paths to any target with the specified cutoff
            path_generator = nx.all_simple_paths(
                G, source=source, target=target_set, cutoff=path_length
            )

            # Process paths, filtering by exact length
            for path in path_generator:
                # if len(path) - 1 == path_length:  # path length is number of edges, which is nodes - 1
                target = path[-1]  # Get the target node from the path

                # Initialize the paths list for this source-target pair if needed
                if (source, target) not in paths_dict:
                    paths_dict[(source, target)] = []

                # Add the path to the results
                paths_dict[(source, target)].append(path)
                source_paths_count += 1
                total_paths_found += 1

        except nx.NetworkXNoPath:
            pass  # No path exists

        source_elapsed = time.time() - source_start_time

    end_time = time.time()
    total_elapsed = end_time - start_time

    # Return timing statistics along with the paths
    stats = {
        "total_time": total_elapsed,
        "total_paths": total_paths_found,
        "source_count": len(source_indices),
        "target_count": len(target_indices),
        "path_length": path_length,
    }

    return paths_dict, stats


# single cell types
sctypes = meta.groupby("cell_type").root_id.nunique()
sctypes = sctypes.index[sctypes == 1]

G = nx.from_scipy_sparse_array(inprop)
sensory_types = meta[(meta.super_class == "sensory") & (~meta.cell_type.isin(sctypes))].cell_type.unique()
dn_types = meta[(meta.super_class == "descending") & (~meta.cell_type.isin(sctypes))].cell_type.unique()

results = []
# set seed
random.seed(42)

selected_sensory_types = random.sample(list(sensory_types), 5)
selected_dn_types = random.sample(list(dn_types), 5)

# ------ networkx ------
# randomly select 5
for length in tqdm([2, 3, 4]):
    for source_type in selected_sensory_types:
        for dn_type in selected_dn_types:
            sources = meta.idx[meta.cell_type == source_type].values
            targets = meta.idx[meta.cell_type == dn_type].values

            _, stats = find_paths_within_length(G, sources, targets, length)

            # Add source count, target count, and path length to stats
            stats["source_count"] = len(sources)
            stats["target_count"] = len(targets)
            stats["path_length"] = length

            results.append(stats)

results_df = pd.DataFrame(results)
results_df.to_csv("pathfinding_networkx_benchmark.csv", index=False)

# ------ find_path_iteratively ------
my_results = []
for length in tqdm([2, 3, 4, 5]):
    for source_type in selected_sensory_types:
        for dn_type in selected_dn_types:
            inidx = meta.idx[meta.cell_type == source_type]
            outidx = meta.idx[meta.cell_type == dn_type]
            # time it 
            start_time = time.time()
            for plength in range(2, length + 1):
                paths = find_path_iteratively(inprop, steps, inidx, outidx, plength)
            end_time = time.time()
            elapsed_time = end_time - start_time
            my_results.append(
                {
                    "source_type": source_type,
                    "dn_type": dn_type,
                    "path_length": length,
                    "total_time": elapsed_time,
                    "source_count": len(inidx),
                    "target_count": len(outidx),
                    "n_connections": len(paths) if paths is not None else 0,
                }
            )

my_results_df = pd.DataFrame(my_results)
my_results_df.to_csv("pathfinding_iteratively_benchmark.csv", index=False)


# ------ find_paths_of_length ------
my_results = []
for length in tqdm([2, 3, 4, 5]):
    for source_type in selected_sensory_types:
        for dn_type in selected_dn_types:
            inidx = meta.idx[meta.cell_type == source_type]
            outidx = meta.idx[meta.cell_type == dn_type]
            # time it
            start_time = time.time()
            for plength in range(2, length + 1):
                paths = find_paths_of_length(inprop, inidx, outidx, plength)
            end_time = time.time()
            elapsed_time = end_time - start_time
            my_results.append(
                {
                    "source_type": source_type,
                    "dn_type": dn_type,
                    "path_length": length,
                    "total_time": elapsed_time,
                    "source_count": len(inidx),
                    "target_count": len(outidx),
                    "n_connections": len(paths) if paths is not None else 0,
                }
            )

my_results_df = pd.DataFrame(my_results)
my_results_df.to_csv("pathfinding_oflength_benchmark.csv", index=False)


# ------ networkit ------
G = nk.Graph(inprop.shape[0], weighted=True, directed=True)
coo = inprop.tocoo()
for u, v, w in zip(coo.row, coo.col, coo.data):
    G.addEdge(int(u), int(v), float(w))
G = nk.graphtools.toUnweighted(G)

def nk_count_paths_fixed_len(G, sources, targets, L):
    import networkit as nk

    total = 0
    for s in sources:
        for t in targets:
            if s == t:
                continue
            asp = nk.reachability.AllSimplePaths(G, int(s), int(t), cutoff=L)
            try:
                asp.run()
            except RuntimeError as e:
                # NetworKit raises if source cannot reach target
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

# 3) benchmark vs your method
nk_results = []
for length in [2,3,4]:
    for source_type in selected_sensory_types:
        for dn_type in selected_dn_types:
            sources = meta.idx[meta.cell_type == source_type].astype(int).values
            targets = meta.idx[meta.cell_type == dn_type].astype(int).values
            t0 = time.time()
            n_paths = nk_count_paths_fixed_len(G, sources, targets, length)
            t1 = time.time()
            nk_results.append({
                "source_type": source_type,
                "dn_type": dn_type,
                "path_length": length,
                "total_time": t1 - t0,
                "source_count": len(sources),
                "target_count": len(targets),
                "n_connections": n_paths,
            })

nk_df = pd.DataFrame(nk_results)
nk_df.to_csv("pathfinding_networkit_benchmark.csv", index=False)