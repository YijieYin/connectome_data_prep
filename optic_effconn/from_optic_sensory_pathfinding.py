import numpy as np
import pandas as pd
import torch
from connectome_interpreter import *
import scipy as sp
from tqdm import tqdm
from pqdm.threads import pqdm

from typing import Iterable, Union, Optional, Callable

# Slurm info
rank = int(os.getenv("SLURM_PROCID", "0"))
world = int(os.getenv("SLURM_NTASKS", "1"))

dataset = 'FAFB' # FAFB or maleCNS 

if dataset == 'FAFB':
    inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_inprop_all_neuron.npz")
    ad_inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_ad_inprop_all_neuron.npz")
    meta = pd.read_csv("../data/fafb_all_neuron/fafb_all_neuron_meta.csv", index_col=0)
elif dataset == 'maleCNS':
    inprop = sp.sparse.load_npz("../data/maleCNS/mcns_inprop_all_neuron.npz")
    meta = pd.read_csv("../data/maleCNS/mcns_all_neuron_meta.csv", index_col=0)
    meta.loc[meta.cell_type.str.contains('^R7') & (meta.cell_type != 'R7d'), 'cell_type'] = 'R7'
    meta.loc[meta.cell_type.str.contains('^R8') & (meta.cell_type != 'R8d'), 'cell_type'] = 'R8'
    meta['side'] = meta.somaSide.fillna(meta.rootSide)

meta['type_side'] = meta.cell_type + "_" + meta.side
idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_type_side = dict(zip(meta.idx, meta.type_side))

# sample unique cell types without replacement
if dataset == 'FAFB':
    pretypes = meta.type_side[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R8']) & (meta.side == 'right') & meta.cell_type.notna()].unique()
    posttypes =meta.type_side[meta.super_class.isin(['optic', 'visual_projection', 'visual_centrifugal']) & (meta.cell_type.notna())].unique()
elif dataset == 'maleCNS':
    pretypes = meta.type_side[meta.cell_type.isin(['L1', 'L2', 'L3', 'R7', 'R8', 'R7d', 'R8d']) & (meta.side == 'R') & meta.cell_type.notna()].unique()
    posttypes =meta.type_side[meta.superclass.isin(['ol_intrinsic', 'visual_projection', 'ol_sensory', 'visual_centrifugal']) & (meta.cell_type.notna())].unique()

# list of pre, post tuples, containing all combinations of pre and post 
prepost = [(pre, post) for pre in pretypes for post in posttypes]

# for this rank: 
my_pairs = np.array_split(prepost, world)[rank]

def process_one(this_set):
    
    pre, post = this_set
    try: 
        outs = []
        for plen in [1, 2, 3, 4, 5]:
            out = {"pre": pre, "post": post, "plen": plen}
            inidx = meta.idx[meta.type_side == pre]
            outidx = meta.idx[meta.type_side == post]
            path = find_paths_of_length(ad_inprop, inidx, outidx, plen)
            if path is not None:
                path = group_paths(path, idx_to_type_side, idx_to_type_side)
                effconn = effective_conn_from_paths(path)
                out['threshold_0'] = effconn.values[0][0]
                for threshold in [0.001, 0.01, 0.03, 0.05, 0.1]:
                    # skip if any layer is completely removed
                    if path[path.weight > threshold].layer.nunique() != path.layer.nunique():
                        out[f'threshold_{threshold}'] = 0
                        continue
                    p_thresh = filter_paths(path, threshold=threshold)
                    if p_thresh is not None:
                        effconn_thresh = effective_conn_from_paths(p_thresh)
                        out[f'threshold_{threshold}'] = effconn_thresh.values[0][0]
                    else:
                        out[f'threshold_{threshold}'] = 0

            else:
                out['threshold_0'] = 0
                for threshold in [0.001, 0.01, 0.03, 0.05, 0.1]:
                    out[f'threshold_{threshold}'] = 0
            outs.append(out)
        out = pd.DataFrame(outs)
        return out
    
    except AssertionError as e:
        import traceback
        print(f"AssertionError for {pre} -> {post}: {e}")
        traceback.print_exc()
        return pd.DataFrame()  # return empty df so concat works

if __name__ == "__main__":
    # 2240 cores available, 5GB per core 
    # 5GB per thread 
    # 20 tasks, 112 threads each
    results = pqdm(my_pairs, process_one, n_jobs=56)
    valid = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
    pd.concat(valid).to_csv(f"results_rank{rank}.csv", index=False)