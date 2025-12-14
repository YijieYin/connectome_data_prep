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
# idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_root = dict(zip(meta.idx, meta.root_id))
idx_to_sign = dict(zip(meta.idx, meta.sign))

inprop_e_only = modify_coo_matrix(inprop, meta.idx[meta.top_nt != 'acetylcholine'], 
                                  meta.idx, 0, re_normalize=False)
ad_inprop_e_only = modify_coo_matrix(ad_inprop, meta.idx[meta.top_nt != 'acetylcholine'], 
                                     meta.idx, 0, re_normalize=False)

threshold = 0
if threshold > 0:
    inprop = sp.sparse.coo_matrix((inprop.data[inprop.data > threshold], 
                                   (inprop.row[inprop.data > threshold], inprop.col[inprop.data > threshold])), 
                                  shape=inprop.shape)
    ad_inprop = sp.sparse.coo_matrix((ad_inprop.data[ad_inprop.data > threshold], 
                                   (ad_inprop.row[ad_inprop.data > threshold], ad_inprop.col[ad_inprop.data > threshold])), 
                                  shape=ad_inprop.shape)
    inprop_e_only = sp.sparse.coo_matrix((inprop_e_only.data[inprop_e_only.data > threshold], 
                                   (inprop_e_only.row[inprop_e_only.data > threshold], inprop_e_only.col[inprop_e_only.data > threshold])), 
                                  shape=inprop_e_only.shape)
    ad_inprop_e_only = sp.sparse.coo_matrix((ad_inprop_e_only.data[ad_inprop_e_only.data > threshold], 
                                   (ad_inprop_e_only.row[ad_inprop_e_only.data > threshold], ad_inprop_e_only.col[ad_inprop_e_only.data > threshold])), 
                                  shape=ad_inprop_e_only.shape)

my_ids = np.array_split(meta.root_id[meta.top_nt.isin(['acetylcholine', 'gaba', 'glutamate'])], world)[rank]

def process_one(pre):
    
    outs = []
    for plen in [2, 3, 4, 5, 6]:
        out = {"pre": pre, "post": pre, "plen": plen}
        inidx = meta.idx[meta.root_id == pre]
        outidx = meta.idx[meta.root_id == pre]
        path = find_paths_of_length(inprop, inidx, outidx, plen)
        if path is not None and (len(path) > 0):
            # path = group_paths(path, idx_to_root, idx_to_root)
            # remove intermediate if it's the same as starting point - only makes sense for single neuron level 
            path = path[((path.layer ==1) | ((path.layer != 1) & (path.pre != pre))) 
                        # OR: last layer, or not-last-layer, and not neuron of interest as post 
                        | ((path.layer == plen) | ((path.layer != plen) & (path.post != pre)))]
            path = remove_excess_neurons(path)
            if path is not None: 
                effconn = effective_conn_from_paths(path)
                e,i = signed_effective_conn_from_paths(path, 
                # to be changed if grouped at type level 
                idx_to_nt = idx_to_sign)
                out['effconn'] = effconn.values[0][0]
                out['effconn_e'] = e.values[0][0]
                out['effconn_i'] = i.values[0][0]
            else: 
                out['effconn'] = 0
                out['effconn_e'] = 0
                out['effconn_i'] = 0
        else:
            out['effconn'] = 0
            out['effconn_e'] = 0
            out['effconn_i'] = 0

        path_e = find_paths_of_length(inprop_e_only, inidx, outidx, plen)
        if path_e is not None and (len(path_e) > 0):
            # path_e = group_paths(path_e, idx_to_root, idx_to_root)
            # remove intermediate if it's the same as starting point 
            path_e = path_e[((path_e.layer ==1) | ((path_e.layer != 1) & (path_e.pre != pre))) 
                        # OR: last layer, or not-last-layer, and not neuron of interest as post 
                        | ((path_e.layer == plen) | ((path_e.layer != plen) & (path_e.post != pre)))]
            path_e = remove_excess_neurons(path_e)
            if path_e is not None: 
                effconn_e = effective_conn_from_paths(path_e)
                out['effconn_e'] = effconn_e.values[0][0]
            else: 
                out['effconn_e'] = 0
        else:
            out['effconn_e'] = 0
        
        path_ad = find_paths_of_length(ad_inprop, inidx, outidx, plen)
        if path_ad is not None and (len(path_ad) > 0):
            # path_ad = group_paths(path_ad, idx_to_root, idx_to_root)
            # remove intermediate if it's the same as starting point 
            path_ad = path_ad[((path_ad.layer ==1) | ((path_ad.layer != 1) & (path_ad.pre != pre))) 
                        # OR: last layer, or not-last-layer, and not neuron of interest as post 
                        | ((path_ad.layer == plen) | ((path_ad.layer != plen) & (path_ad.post != pre)))]
            path_ad = remove_excess_neurons(path_ad)
            if path_ad is not None: 
                effconn_ad = effective_conn_from_paths(path_ad)
                e,i = signed_effective_conn_from_paths(path_ad, idx_to_nt = idx_to_sign)
                out['effconn_ad'] = effconn_ad.values[0][0]
                out['effconn_ad_e'] = e.values[0][0]
                out['effconn_ad_i'] = i.values[0][0]
            else: 
                out['effconn_ad'] = 0
                out['effconn_ad_e'] = 0
                out['effconn_ad_i'] = 0
        else:
            out['effconn_ad'] = 0
            out['effconn_ad_e'] = 0
            out['effconn_ad_i'] = 0

        path_ad_e = find_paths_of_length(ad_inprop_e_only, inidx, outidx, plen)
        if path_ad_e is not None and (len(path_ad_e) > 0):
            # path_ad_e = group_paths(path_ad_e, idx_to_root, idx_to_root)
            # remove intermediate if it's the same as starting point 
            path_ad_e = path_ad_e[((path_ad_e.layer ==1) | ((path_ad_e.layer != 1) & (path_ad_e.pre != pre))) 
                        # OR: last layer, or not-last-layer, and not neuron of interest as post 
                        | ((path_ad_e.layer == plen) | ((path_ad_e.layer != plen) & (path_ad_e.post != pre)))]
            path_ad_e = remove_excess_neurons(path_ad_e)
            if path_ad_e is not None: 
                effconn_ad_e = effective_conn_from_paths(path_ad_e)
                out['effconn_ad_e'] = effconn_ad_e.values[0][0]
            else:
                out['effconn_ad_e'] = 0
        else:
            out['effconn_ad_e'] = 0
        outs.append(out)
    out = pd.DataFrame(outs)  
    return out

if __name__ == "__main__":
    results = pqdm(my_ids, process_one, n_jobs=28)

    pd.concat(results).to_csv(f"sc_recurrence_results_rank{rank}.csv", index=False)
