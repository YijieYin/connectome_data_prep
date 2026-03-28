import navis
import os 
import pandas as pd
from scipy.spatial import cKDTree
from pqdm.processes import pqdm

prefix = "720575940"
folder = '/cephfs2/yyin/ad_split'
conn_columns = ["connector_id", "node_id", "type", "x", "y", "z", "neuropil", 'pre_root_id_720575940','post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']
sk_path = os.path.join(folder, "sk_lod1_783_healed")
skids = os.listdir(sk_path)

syn = pd.read_csv(os.path.join(folder, "fafb_v783_princeton_synapse_table.csv"))
# remove autapses 
syn = syn[syn.pre_root_id_720575940 != syn.post_root_id_720575940].drop(columns=['ctr_x', 'ctr_y', 'ctr_z', 'size']).drop_duplicates()

meta = pd.read_csv('https://raw.githubusercontent.com/YijieYin/connectome_data_prep/refs/heads/main/data/fafb_all_neuron/fafb_all_neuron_meta.csv', index_col=0)
meta.loc[:, ["root_id_short"]] = meta.root_id.apply(
    lambda x: int(str(x).split(prefix)[1])
)

selected_skids = set(meta.root_id_short[
    (meta.super_class.isin(
        [
            "optic",
            "central",
            "visual_projection",
            "visual_centrifugal",
            "descending",
            "endocrine",
        ]
    ) & (meta.cell_class != 'Kenyon_Cell') & (~ meta.cell_type.isin(['APL','DPM']))) 
    | (meta.cell_type == 'CB0769') # the only motor neuron that has a bit of axon 
])

# processed = os.listdir(os.path.join(folder, "sk_783_with_connectors"))
# # remove prefix and '.csv'
# processed = set([int(f.split(prefix)[1].split(".")[0]) for f in processed])
# selected_skids = selected_skids - processed
# print(len(selected_skids), "skids to process")

def split_one(sk): 
    # for sk in {616959010,624167973}:
    n = navis.read_swc(os.path.join(sk_path, prefix + str(sk) + ".swc"))

    shortid = int(n.name.split(prefix)[1])
    tree = cKDTree(n.nodes[["x", "y", "z"]])

    syn_pre = syn[syn.pre_root_id_720575940 == shortid]
    if len(syn_pre) == 0:
        # make empty dataframe with columns:
        syn_pre = pd.DataFrame(columns=conn_columns)
    else:
        # find the closest node for each synapse ----
        d, idx = tree.query(syn_pre[["pre_x", "pre_y", "pre_z"]])
        syn_pre.loc[:, ["node_id"]] = n.nodes.loc[idx, :].node_id.values
        syn_pre.loc[:, ["type"]] = "pre"
        syn_pre = syn_pre.rename(
            columns={
                "pre_x": "x",
                "pre_y": "y",
                "pre_z": "z",
                'post_x': 'partner_x',
                'post_y': 'partner_y',
                'post_z': 'partner_z',
            },
        )
        syn_pre.loc[:, ["connector_id"]] = range(len(syn_pre))

    syn_post = syn[syn.post_root_id_720575940 == shortid]
    if len(syn_post) == 0:
        # make empty dataframe with columns:
        syn_post = pd.DataFrame(columns=conn_columns)
    else:
        d, idx = tree.query(syn_post[["post_x", "post_y", "post_z"]])
        syn_post.loc[:, ["node_id"]] = n.nodes.loc[idx, :].node_id.values
        syn_post.loc[:, ["type"]] = "post"
        syn_post = syn_post.rename(
            columns={
                "post_x": "x",
                "post_y": "y",
                "post_z": "z",
                'pre_x': 'partner_x',
                'pre_y': 'partner_y',
                'pre_z': 'partner_z',
            },
        )
        syn_post.loc[:, ["connector_id"]] = range(
            len(syn_pre), len(syn_pre) + len(syn_post)
        )

    n.connectors = pd.concat([syn_pre[conn_columns], syn_post[conn_columns]], ignore_index=True)
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
        n, os.path.join(folder, "sk_783_with_connectors"), 
        export_connectors = True
    )
    del n, syn_pre, syn_post, tree
    import gc; gc.collect()

if __name__ == "__main__":
    pqdm(selected_skids, split_one, n_jobs=int(os.getenv("SLURM_NTASKS", 1)))

