import navis
import os 
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pqdm.processes import pqdm

# Slurm info
rank = int(os.getenv("SLURM_PROCID", "0"))
world = int(os.getenv("SLURM_NTASKS", "1"))

folder = '/cephfs2/yyin/mcns_ad_split'
conn_columns = ["connector_id", "node_id", "type", "x", "y", "z", "neuropil", 'body_pre','body_post', 'partner_x', 'partner_y', 'partner_z']
sk_path = os.path.join(folder, "skeletons-swc")
skids = os.listdir(sk_path)

meta = pd.read_csv('data/maleCNS/mcns_all_neuron_meta.csv', index_col=0)
meta = meta.rename(columns={"superclass": "super_class", 'class': 'cell_class', 'subclass': 'cell_sub_class'})
id2type = dict(zip(meta.bodyid, meta.cell_type))

syn = pd.read_feather(os.path.join(folder, "syn-partners-male-cns-v1.0-minconf-0.5.feather"))
# remove autapses 
syn = syn[syn.body_pre != syn.body_post]
# and only keep those with meta 
syn = syn[syn.body_pre.isin(id2type) & syn.body_post.isin(id2type)].rename(columns={"primary_post": "neuropil"}).copy()

selected_skids = set(meta.bodyid[
    (~meta.super_class.str.contains('sensory|motor')) 
    & (meta.cell_class != 'Kenyon_Cell') 
    & (~ meta.cell_type.isin(['APL','DPM'])) 
])

# processed = os.listdir(os.path.join(folder, "sk_with_connectors"))
# # remove prefix and '.csv'
# processed = set([int(f.split(".")[0]) for f in processed])
# selected_skids = selected_skids - processed
# print(len(selected_skids), "skids to process")

# make the relevant folders if not exist
# `split_failures` gets one csv per rank listing the neurons that error out
for subfolder in ["axon_in", "axon_out", "dendrite_in", "dendrite_out", "seg_indices", "sk_with_connectors", "split_failures"]:
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

def split_one(sk):
    # for sk in {616959010,624167973}:
    n = navis.read_swc(os.path.join(sk_path, str(sk) + ".swc"))

    # navis.split_axon_dendrite() refuses neurons with >1 root, so stitch
    # disconnected fragments back together first
    if len(n.root) > 1:
        n = navis.heal_skeleton(n, method="ALL")

    tree = cKDTree(n.nodes[["x", "y", "z"]])

    syn_pre = syn[syn.body_pre == sk]
    if len(syn_pre) == 0:
        # make empty dataframe with columns:
        syn_pre = pd.DataFrame(columns=conn_columns)
    else:
        # find the closest node for each synapse ----
        d, idx = tree.query(syn_pre[["x_pre", "y_pre", "z_pre"]])
        syn_pre.loc[:, ["node_id"]] = n.nodes.loc[idx, :].node_id.values
        syn_pre.loc[:, ["type"]] = "pre"
        syn_pre = syn_pre.rename(
            columns={
                "x_pre": "x",
                "y_pre": "y",
                "z_pre": "z",
                'x_post': 'partner_x',
                'y_post': 'partner_y',
                'z_post': 'partner_z',
            },
        )
        syn_pre.loc[:, ["connector_id"]] = range(len(syn_pre))

    syn_post = syn[syn.body_post == sk]
    if len(syn_post) == 0:
        # make empty dataframe with columns:
        syn_post = pd.DataFrame(columns=conn_columns)
    else:
        d, idx = tree.query(syn_post[["x_post", "y_post", "z_post"]])
        syn_post.loc[:, ["node_id"]] = n.nodes.loc[idx, :].node_id.values
        syn_post.loc[:, ["type"]] = "post"
        syn_post = syn_post.rename(
            columns={
                "x_post": "x",
                "y_post": "y",
                "z_post": "z",
                'x_pre': 'partner_x',
                'y_pre': 'partner_y',
                'z_pre': 'partner_z',
            },
        )
        syn_post.loc[:, ["connector_id"]] = range(
            len(syn_pre), len(syn_pre) + len(syn_post)
        )

    n.connectors = pd.concat([syn_pre[conn_columns], syn_post[conn_columns]], ignore_index=True)

    # navis needs both pre- and postsynapses to split a neuron and to compute its
    # segregation index; manual_split.py makes these cell types unsplit instead
    if n.connectors.type.nunique() < 2:
        return

    n = navis.split_axon_dendrite(n, label_only=True)

    # note: linker treated as axon 
    n.connectors[
        (n.connectors.compartment.isin(['axon','linker'])) & (n.connectors.type == "pre")
    ].drop(columns=["connector_id", "node_id", "type", "compartment"]).to_csv(
            os.path.join(folder, f"axon_out/{n.name}.csv"), index=False
    )
    n.connectors[
        (n.connectors.compartment == "dendrite") & (n.connectors.type == "pre")
    ].drop(columns=["connector_id", "node_id", "type", "compartment"]).to_csv(
        os.path.join(folder, f"dendrite_out/{n.name}.csv"), index=False
    )
    n.connectors[
        (n.connectors.compartment.isin(['axon','linker'])) & (n.connectors.type == "post")
    ].drop(columns=["connector_id", "node_id", "type", "compartment"]).to_csv(
        os.path.join(folder, f"axon_in/{n.name}.csv"), index=False)

    n.connectors[
        (n.connectors.compartment == "dendrite") & (n.connectors.type == "post")
    ].drop(columns=["connector_id", "node_id", "type", "compartment"]).to_csv(
        os.path.join(folder, f"dendrite_in/{n.name}.csv"), index=False
    )

    _ = navis.arbor_segregation_index(n)
    seg_df = pd.DataFrame({
        "root_id": [n.name],
        "segregation_index": [n.nodes.segregation_index.max()]
    })
    seg_df.to_csv(os.path.join(folder, "seg_indices", f"{n.name}.csv"), index=False)

    # save with connectors
    navis.write_swc(
        n, os.path.join(folder, "sk_with_connectors"), 
        export_connectors = True
    )
    del n, syn_pre, syn_post, tree
    import gc; gc.collect()

if __name__ == "__main__":
    all_ids = list(selected_skids)
    my_ids = np.array_split(all_ids, world)[rank]

    # pqdm returns the exception instead of raising it, so log the failures
    results = pqdm(my_ids, split_one, n_jobs=56)
    failed = [(sk, repr(r)) for sk, r in zip(my_ids, results) if isinstance(r, Exception)]
    if failed:
        print(f"{len(failed)} of {len(my_ids)} neurons failed, see split_failures/rank{rank}.csv")
        pd.DataFrame(failed, columns=["bodyid", "error"]).to_csv(
            os.path.join(folder, "split_failures", f"rank{rank}.csv"), index=False
        )