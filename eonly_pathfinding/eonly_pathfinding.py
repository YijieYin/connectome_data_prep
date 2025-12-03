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

inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_inprop_all_neuron.npz")
meta = pd.read_csv("../data/fafb_all_neuron/fafb_all_neuron_meta.csv")
idx_to_type = dict(zip(meta.idx, meta.cell_type))

inprop_e_only = modify_coo_matrix(inprop, meta.idx[meta.sign == -1], 
                                  meta.idx, 0)

npre = 1000
npost = 1000

# set NumPy seed
np.random.seed(42)

# sample unique cell types without replacement
valid_types = meta.cell_type[~meta.cell_type.str.isnumeric()].unique()
pretypes = np.random.choice(valid_types, npre, replace=False)
posttypes = np.random.choice(valid_types, npost, replace=False)

my_pretypes = np.array_split(pretypes, world)[rank]

def process_one(this_set):
    
    pre = this_set
    outs = []
    for post in posttypes:
        for plen in [2, 3, 4, 5]:
            out = {"pre": pre, "post": post, "plen": plen}
            inidx = meta.idx[meta.cell_type == pre]
            outidx = meta.idx[meta.cell_type == post]
            path = find_paths_of_length(inprop_e_only, inidx, outidx, plen)
            if path is not None:
                path = group_paths(path, idx_to_type, idx_to_type)
                out['threshold_0'] = True
                for threshold in [0.001, 0.01, 0.03, 0.05, 0.1]:
                    # skip if any layer is completely removed
                    if path[path.weight > threshold].layer.nunique() != path.layer.nunique():
                        out[f'threshold_{threshold}'] = False
                        continue
                    p_thresh = filter_paths(path, threshold=threshold)
                    if p_thresh is not None:
                        out[f'threshold_{threshold}'] = True
                    else: 
                        out[f'threshold_{threshold}'] = False

            else:
                out['threshold_0'] = False
                for threshold in [0.001, 0.01, 0.03, 0.05, 0.1]:
                    out[f'threshold_{threshold}'] = False
            outs.append(out)
    out = pd.DataFrame(outs)
    return out

if __name__ == "__main__":
    results = pqdm(my_pretypes, process_one, n_jobs=50)
    pd.concat(results).to_csv(f"eonly_pathfinding_results_rank{rank}.csv", index=False)

