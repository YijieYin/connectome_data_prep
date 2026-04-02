import argparse
import numpy as np
import pandas as pd
import torch
from connectome_interpreter import *
import scipy as sp
from pqdm.threads import pqdm

from typing import Iterable, Union, Optional, Callable

p = argparse.ArgumentParser(description="")
p.add_argument("--count-paths", action=argparse.BooleanOptionalAction, default=False,
                   help="Whether to count enumerated paths (time-consuming)")
p.add_argument("--effconn-noloop", action=argparse.BooleanOptionalAction, default=False,
                   help="Whether to calculate effective connectivity without loops (time-consuming)")
p.add_argument("--effconn", action=argparse.BooleanOptionalAction, default=False,
                   help="Whether to calculate effective connectivity")
args = p.parse_args()

# Slurm info
rank = int(os.getenv("SLURM_PROCID", "0"))
world = int(os.getenv("SLURM_NTASKS", "1"))

# bind GPU per task (CUDA_VISIBLE_DEVICES already narrowed by Slurm)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# make sure you choose the one you want (inprop/ad_inprop) in find_paths_of_length(). 
# and make sure you change the file name at the end and in concat_result.py accordingly.
inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_inprop_all_neuron.npz")
ad_inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_ad_inprop_all_neuron.npz")
meta = pd.read_csv("../data/fafb_all_neuron/fafb_all_neuron_meta.csv")
meta['type_side'] = meta['cell_type'] + "_" + meta['side']
idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_type_side = dict(zip(meta.idx, meta.type_side))

# remove 1 synapse connections ----
ad_syncount = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_ad_syncount_all_neuron.npz").tocoo()
onesyn = pd.DataFrame({
    'input_idx': ad_syncount.row[ad_syncount.data == 1],
    'output_idx': ad_syncount.col[ad_syncount.data == 1],
    'value': 0
})
ad_inprop = modify_coo_matrix(ad_inprop, updates_df=onesyn)

# thresholding ---- 
threshold = 0
if threshold > 0:
    update_df = pd.DataFrame({
        'input_idx': inprop.row[inprop.data < threshold],
        'output_idx': inprop.col[inprop.data < threshold],
        'value': 0
    })
    inprop = modify_coo_matrix(inprop, updates_df=update_df)

# sensory to DNs 
pretypes = meta.type_side[~meta.cell_type.str.isnumeric() & (meta.super_class == 'sensory')].unique()
posttypes = meta.type_side[~meta.cell_type.str.isnumeric() & (meta.super_class == 'descending')].unique()

# alternative random selection ----  
# npre = 1000
# npost = 1000

# # set NumPy seed
# np.random.seed(42)

# # sample unique cell types without replacement
# valid_types = meta.type_side[~meta.cell_type.str.isnumeric()].unique()
# pretypes = np.random.choice(valid_types, npre, replace=False)
# posttypes = np.random.choice(valid_types, npost, replace=False)

# list of pre, post tuples, containing all combinations of pre and post 
prepost = [(pre, post) for pre in pretypes for post in posttypes]

# for this rank: 
my_pairs = np.array_split(prepost, world)[rank]

thresholds = [0.001, 0.008, 0.009, 0.01, 0.012, 0.015, 0.03, 0.05, 0.1]

def process_one(this_set):
    
    pre, post = this_set
    outs = []
    
    for plen in [2, 3, 4, 5]:
        out = {"pre": pre, "post": post, "plen": plen}
        inidx = meta.idx[meta.type_side == pre]
        outidx = meta.idx[meta.type_side == post]
        path = find_paths_of_length(ad_inprop, inidx, outidx, plen)
        if path is not None:
            path = group_paths(path, idx_to_type_side, idx_to_type_side)
            out['threshold_0'] = True
            if args.effconn:
                out['effconn_0'] = effective_conn_from_paths(path).values[0][0]
            if args.count_paths: 
                n_paths, n_paths_noloop = count_paths(path, loop_mode = 'both')
                out['n_paths'] = n_paths
                out['n_paths_noloop'] = n_paths_noloop
            for threshold in thresholds:
                # skip if any layer is completely removed
                if path[path.weight > threshold].layer.nunique() != path.layer.nunique():
                    out[f'threshold_{threshold}'] = False
                    continue
                p_thresh = filter_paths(path, threshold=threshold)
                if p_thresh is not None:
                    out[f'threshold_{threshold}'] = True
                    # calculate effective connectivity for these thresholds 
                    if args.effconn:
                        if threshold in [0.01, 0.05]: 
                            out[f'effconn_{threshold}'] = effective_conn_from_paths(p_thresh).values[0][0]
                else: 
                    out[f'threshold_{threshold}'] = False

            if args.effconn_noloop:
                effconn_noloop = effconn_without_loops(path, quiet = True)
                out['effconn_noloop'] = effconn_noloop.values[0][0]

        else:
            out['threshold_0'] = False
            if args.effconn:
                out['effconn_0'] = 0
            if args.count_paths: 
                out['n_paths'] = 0
                out['n_paths_noloop'] = 0
            for threshold in thresholds:
                out[f'threshold_{threshold}'] = False
                if args.effconn:
                    if threshold in [0.01, 0.05]: 
                        out[f'effconn_{threshold}'] = 0
            if args.effconn_noloop:
                out['effconn_noloop'] = None
            
        outs.append(out)
    out = pd.DataFrame(outs)
    return out 

if __name__ == "__main__":
    # 2240 cores available, 5GB per core 
    # 5GB per thread 
    # 20 tasks, 112 threads each
    results = pqdm(my_pairs, process_one, n_jobs=112, )
    
    # Save dataframes - NOTE also need to change the file name in concat_result.py
    pd.concat(results).to_csv(f"ad_sensory_dn_thresholds_onesyn_rank{rank}.csv", index=False)
