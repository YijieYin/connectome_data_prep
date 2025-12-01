import numpy as np
import pandas as pd
from connectome_interpreter import *
import scipy as sp
from pqdm.threads import pqdm

from typing import Iterable, Union, Optional, Callable

# Slurm info
rank = int(os.getenv("SLURM_PROCID", "0"))
world = int(os.getenv("SLURM_NTASKS", "1"))

inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_inprop_all_neuron.npz")
ad_inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_ad_inprop_all_neuron.npz")
meta = pd.read_csv("../data/fafb_all_neuron/fafb_all_neuron_meta.csv")
idx_to_type = dict(zip(meta.idx, meta.cell_type))

inprop_e_only = modify_coo_matrix(inprop, meta.idx[meta.sign == -1], 
                                  meta.idx, 0, re_normalize=False)
ad_inprop_e_only = modify_coo_matrix(ad_inprop, meta.idx[meta.sign == -1], 
                                     meta.idx, 0, re_normalize=False)

# sample unique cell types without replacement
valid_types = meta.cell_type[~meta.cell_type.str.isnumeric()].unique()
my_pretypes = np.array_split(valid_types, world)[rank]

print(f"Rank {rank}/{world}: {len(my_pretypes)} types")

def process_one(pre):
    
    outs = []
    for plen in [1, 2, 3, 4, 5]:
        out = {"pre": pre, "post": pre, "plen": plen}
        inidx = meta.idx[meta.cell_type == pre]
        outidx = meta.idx[meta.cell_type == pre]
        path = find_paths_of_length(inprop, inidx, outidx, plen)
        if path is not None and (len(path) > 0):
            path = group_paths(path, idx_to_type, idx_to_type)
            out['connected'] = True
            effconn = effective_conn_from_paths(path)
            out['effconn'] = effconn.values[0][0]
        else:
            out['connected'] = False
            out['effconn'] = 0

        path_e = find_paths_of_length(inprop_e_only, inidx, outidx, plen)
        if path_e is not None and (len(path_e) > 0):
            path_e = group_paths(path_e, idx_to_type, idx_to_type)
            effconn_e = effective_conn_from_paths(path_e)
            out['effconn_e'] = effconn_e.values[0][0]
        else:
            out['effconn_e'] = 0
        
        path_ad = find_paths_of_length(ad_inprop, inidx, outidx, plen)
        if path_ad is not None and (len(path_ad) > 0):
            path_ad = group_paths(path_ad, idx_to_type, idx_to_type)
            effconn_ad = effective_conn_from_paths(path_ad)
            out['effconn_ad'] = effconn_ad.values[0][0]
        else:
            out['effconn_ad'] = 0

        path_ad_e = find_paths_of_length(ad_inprop_e_only, inidx, outidx, plen)
        if path_ad_e is not None and (len(path_ad_e) > 0):
            path_ad_e = group_paths(path_ad_e, idx_to_type, idx_to_type)
            effconn_ad_e = effective_conn_from_paths(path_ad_e)
            out['effconn_ad_e'] = effconn_ad_e.values[0][0]
        else:
            out['effconn_ad_e'] = 0
        outs.append(out)
    out = pd.DataFrame(outs)  
    return out

if __name__ == "__main__":
    results = pqdm(my_pretypes, process_one, n_jobs=28)

    pd.concat(results).to_csv(f"recurrence_results_rank{rank}.csv", index=False)
