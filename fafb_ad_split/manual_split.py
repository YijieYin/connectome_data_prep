import os 
import pandas as pd
import numpy as np
from tqdm import tqdm

prefix = "720575940"
folder = '/cephfs2/yyin/ad_split'

sk_path = os.path.join(folder, "sk_lod1_783_healed")
skids = os.listdir(sk_path)

meta = pd.read_csv('https://raw.githubusercontent.com/YijieYin/connectome_data_prep/refs/heads/main/data/fafb_all_neuron/fafb_all_neuron_meta.csv', index_col=0)
meta.loc[:, ["root_id_short"]] = meta.root_id.apply(
    lambda x: int(str(x).split(prefix)[1])
)
id2type = dict(zip(meta.root_id, meta.cell_type))

syn = pd.read_csv(os.path.join(folder, "fafb_v783_princeton_synapse_table.csv"))
# remove autapses 
syn = syn[syn.pre_root_id_720575940 != syn.post_root_id_720575940]

# ---- KC ---- 
kc_shortids = meta.loc[meta.cell_class == "Kenyon_Cell", "root_id_short"].values
kc_syns = syn[syn.pre_root_id_720575940.isin(kc_shortids) | syn.post_root_id_720575940.isin(kc_shortids)]
kc_syns.loc[:,['compartment']] = kc_syns.neuropil.apply(
    lambda x: "dendrite" if x in ["MB_CA_L", 'MB_CA_R', "LH_L", 'LH_R', "PLP_L", 'PLP_R',
                                  "SCL_L", 'SCL_R', "ICL_L", "ICL_R", "SLP_L", 'SLP_R', 
                                  "PB", "LO_L", 'LO_R', "ATL_L", 'ATL_R'] else "axon"
)
for kcid in tqdm(meta.root_id_short[meta.cell_class == "Kenyon_Cell"].values):
    axon_out = kc_syns.loc[(kc_syns.pre_root_id_720575940 == kcid) & (kc_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(kcid)}.csv"), index=False)
    axon_in = kc_syns.loc[(kc_syns.post_root_id_720575940 == kcid) & (kc_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(kcid)}.csv"), index=False)
    dendrite_out = kc_syns.loc[(kc_syns.pre_root_id_720575940 == kcid) & (kc_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(kcid)}.csv"), index=False)
    dendrite_in = kc_syns.loc[(kc_syns.post_root_id_720575940 == kcid) & (kc_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(kcid)}.csv"), index=False)


# ---- tangential neurons ---- 
tan_shortids = meta.loc[(meta.cell_sub_class == "tangential") 
& (meta.super_class == 'central') 
# FB intrinsic types, not spliting 
& (~meta.cell_type.isin(['FB4Z', 'FB5R,FB5U', 'FB5S', 'FB6,FB6J', 'FB4Z', 'FB6L', 'FB7D', 'FB7J'])), 
"root_id_short"].values
tan_syns = syn[syn.pre_root_id_720575940.isin(tan_shortids) | syn.post_root_id_720575940.isin(tan_shortids)]

tan_syns.loc[:,['compartment']] = tan_syns.neuropil.apply(
    lambda x: "axon" if x in ["FB",'NO'] else "dendrite"
)

for tanid in tqdm(tan_shortids):
    axon_out = tan_syns.loc[(tan_syns.pre_root_id_720575940 == tanid) & (tan_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(tanid)}.csv"), index=False)
    axon_in = tan_syns.loc[(tan_syns.post_root_id_720575940 == tanid) & (tan_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(tanid)}.csv"), index=False)
    dendrite_out = tan_syns.loc[(tan_syns.pre_root_id_720575940 == tanid) & (tan_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(tanid)}.csv"), index=False)
    dendrite_in = tan_syns.loc[(tan_syns.post_root_id_720575940 == tanid) & (tan_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(tanid)}.csv"), index=False)


# ---- FR, FS, FC ---- 
frsc_shortids = meta.loc[meta.cell_type.str.contains('^FR') | meta.cell_type.str.contains('FS') | meta.cell_type.str.contains('FC'), "root_id_short"].values
frsc_syns = syn[syn.pre_root_id_720575940.isin(frsc_shortids) | syn.post_root_id_720575940.isin(frsc_shortids)]

frsc_syns.loc[:,['compartment']] = frsc_syns.neuropil.apply(
    lambda x: "dendrite" if x in ["FB"] else "axon"
)
for frscid in tqdm(frsc_shortids):
    axon_out = frsc_syns.loc[(frsc_syns.pre_root_id_720575940 == frscid) & (frsc_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(frscid)}.csv"), index=False)
    axon_in = frsc_syns.loc[(frsc_syns.post_root_id_720575940 == frscid) & (frsc_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(frscid)}.csv"), index=False)
    dendrite_out = frsc_syns.loc[(frsc_syns.pre_root_id_720575940 == frscid) & (frsc_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(frscid)}.csv"), index=False)
    dendrite_in = frsc_syns.loc[(frsc_syns.post_root_id_720575940 == frscid) & (frsc_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(frscid)}.csv"), index=False)


# ---- SA1-3 ---- 
sa123_shortids = meta.loc[meta.cell_type.isin(['SA1', 'SA2', 'SA3']), 'root_id_short'].values
sa123_syns = syn[syn.pre_root_id_720575940.isin(sa123_shortids) | syn.post_root_id_720575940.isin(sa123_shortids)]
sa123_syns.loc[:,['compartment']] = sa123_syns.neuropil.apply(
    lambda x: "axon" if x in ["FB", 'NO'] else "dendrite"
)
for sa123id in tqdm(sa123_shortids):
    axon_out = sa123_syns.loc[(sa123_syns.pre_root_id_720575940 == sa123id) & (sa123_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(sa123id)}.csv"), index=False)
    axon_in = sa123_syns.loc[(sa123_syns.post_root_id_720575940 == sa123id) & (sa123_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(sa123id)}.csv"), index=False)
    dendrite_out = sa123_syns.loc[(sa123_syns.pre_root_id_720575940 == sa123id) & (sa123_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(sa123id)}.csv"), index=False)
    dendrite_in = sa123_syns.loc[(sa123_syns.post_root_id_720575940 == sa123id) & (sa123_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(sa123id)}.csv"), index=False)

# ---- ExR5 ---- 
exr5_shortids = meta.loc[meta.cell_type == 'ExR5', "root_id_short"].values
exr5_syns = syn[syn.pre_root_id_720575940.isin(exr5_shortids) | syn.post_root_id_720575940.isin(exr5_shortids)]
exr5_syns.loc[:,['compartment']] = exr5_syns.neuropil.apply(
    lambda x: "dendrite" if x in ["SPS_L", 'SPS_R', 'IB_L', 'IB_R', 'ICL_L', 'ICL_R', 'PLP_R', 'PLP_L', 'IPS_R', 'IPS_L', 'ATL_R', 'ATL_L', 'PB'] else "axon"
)
for exr5id in tqdm(exr5_shortids):
    axon_out = exr5_syns.loc[(exr5_syns.pre_root_id_720575940 == exr5id) & (exr5_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(exr5id)}.csv"), index=False)
    axon_in = exr5_syns.loc[(exr5_syns.post_root_id_720575940 == exr5id) & (exr5_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(exr5id)}.csv"), index=False)
    dendrite_out = exr5_syns.loc[(exr5_syns.pre_root_id_720575940 == exr5id) & (exr5_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(exr5id)}.csv"), index=False)
    dendrite_in = exr5_syns.loc[(exr5_syns.post_root_id_720575940 == exr5id) & (exr5_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(exr5id)}.csv"), index=False)

# ---- ExR1 ----
exr1_shortids = meta.loc[meta.cell_type == 'ExR1', "root_id_short"].values
exr1_syns = syn[syn.pre_root_id_720575940.isin(exr1_shortids) | syn.post_root_id_720575940.isin(exr1_shortids)]

exr1_syns.loc[:,['compartment']] = exr1_syns.neuropil.apply(
    lambda x: "axon" if x in ["EB"] else "dendrite"
)
for exr1id in tqdm(exr1_shortids):
    axon_out = exr1_syns.loc[(exr1_syns.pre_root_id_720575940 == exr1id) & (exr1_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(exr1id)}.csv"), index=False)
    axon_in = exr1_syns.loc[(exr1_syns.post_root_id_720575940 == exr1id) & (exr1_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(exr1id)}.csv"), index=False)
    dendrite_out = exr1_syns.loc[(exr1_syns.pre_root_id_720575940 == exr1id) & (exr1_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(exr1id)}.csv"), index=False)
    dendrite_in = exr1_syns.loc[(exr1_syns.post_root_id_720575940 == exr1id) & (exr1_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(exr1id)}.csv"), index=False)

# ---- ring neurons ---- 
ring_shortids = meta.loc[meta.cell_sub_class == 'ring neuron', "root_id_short"].values
ring_syns = syn[syn.pre_root_id_720575940.isin(ring_shortids) | syn.post_root_id_720575940.isin(ring_shortids)]
ring_syns.loc[:,['compartment']] = ring_syns.neuropil.apply(
    lambda x: "axon" if x in ['EB'] else "dendrite"
)
for ringid in tqdm(ring_shortids):
    axon_out = ring_syns.loc[(ring_syns.pre_root_id_720575940 == ringid) & (ring_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(ringid)}.csv"), index=False)
    axon_in = ring_syns.loc[(ring_syns.post_root_id_720575940 == ringid) & (ring_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(ringid)}.csv"), index=False)
    dendrite_out = ring_syns.loc[(ring_syns.pre_root_id_720575940 == ringid) & (ring_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(ringid)}.csv"), index=False)
    dendrite_in = ring_syns.loc[(ring_syns.post_root_id_720575940 == ringid) & (ring_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(ringid)}.csv"), index=False)

# ---- PFN ----
pfn_shortids = meta.loc[meta.cell_type.str.contains('PFN'), "root_id_short"].values
pfn_syns = syn[syn.pre_root_id_720575940.isin(pfn_shortids) | syn.post_root_id_720575940.isin(pfn_shortids)]

pfn_syns.loc[:,['compartment']] = pfn_syns.neuropil.apply(
    lambda x: "axon" if x in ['FB'] else "dendrite"
)

for pfnid in tqdm(pfn_shortids):
    axon_out = pfn_syns.loc[(pfn_syns.pre_root_id_720575940 == pfnid) & (pfn_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(pfnid)}.csv"), index=False)
    axon_in = pfn_syns.loc[(pfn_syns.post_root_id_720575940 == pfnid) & (pfn_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(pfnid)}.csv"), index=False)
    dendrite_out = pfn_syns.loc[(pfn_syns.pre_root_id_720575940 == pfnid) & (pfn_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(pfnid)}.csv"), index=False)
    dendrite_in = pfn_syns.loc[(pfn_syns.post_root_id_720575940 == pfnid) & (pfn_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(pfnid)}.csv"), index=False)

# ---- PEN ---- 
pen_shortids = meta.loc[meta.cell_type.str.contains('PEN'), "root_id_short"].values
pen_syns = syn[syn.pre_root_id_720575940.isin(pen_shortids) | syn.post_root_id_720575940.isin(pen_shortids)]

pen_syns.loc[:,['compartment']] = pen_syns.neuropil.apply(
    lambda x: "axon" if x in ['EB'] else "dendrite"
)

for penid in tqdm(pen_shortids):
    axon_out = pen_syns.loc[(pen_syns.pre_root_id_720575940 == penid) & (pen_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(penid)}.csv"), index=False)
    axon_in = pen_syns.loc[(pen_syns.post_root_id_720575940 == penid) & (pen_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(penid)}.csv"), index=False)
    dendrite_out = pen_syns.loc[(pen_syns.pre_root_id_720575940 == penid) & (pen_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(penid)}.csv"), index=False)
    dendrite_in = pen_syns.loc[(pen_syns.post_root_id_720575940 == penid) & (pen_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(penid)}.csv"), index=False)

# ---- EPG ----
epg_shortids = meta.loc[meta.cell_type == 'EPG', "root_id_short"].values
epg_syns = syn[syn.pre_root_id_720575940.isin(epg_shortids) | syn.post_root_id_720575940.isin(epg_shortids)]

epg_syns.loc[:,['compartment']] = epg_syns.neuropil.apply(
    lambda x: "dendrite" if x in ['EB'] else "axon"
)

for epgid in tqdm(epg_shortids):
    axon_out = epg_syns.loc[(epg_syns.pre_root_id_720575940 == epgid) & (epg_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(epgid)}.csv"), index=False)
    axon_in = epg_syns.loc[(epg_syns.post_root_id_720575940 == epgid) & (epg_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(epgid)}.csv"), index=False)
    dendrite_out = epg_syns.loc[(epg_syns.pre_root_id_720575940 == epgid) & (epg_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(epgid)}.csv"), index=False)
    dendrite_in = epg_syns.loc[(epg_syns.post_root_id_720575940 == epgid) & (epg_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(epgid)}.csv"), index=False)

# ---- PEG ----
peg_shortids = meta.loc[meta.cell_type.str.contains('PEG'), "root_id_short"].values
peg_syns = syn[syn.pre_root_id_720575940.isin(peg_shortids) | syn.post_root_id_720575940.isin(peg_shortids)]

peg_syns.loc[:,['compartment']] = peg_syns.neuropil.apply(
    lambda x: "dendrite" if x in ['PB'] else "axon"
)

for pegid in tqdm(peg_shortids):
    axon_out = peg_syns.loc[(peg_syns.pre_root_id_720575940 == pegid) & (peg_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(pegid)}.csv"), index=False)
    axon_in = peg_syns.loc[(peg_syns.post_root_id_720575940 == pegid) & (peg_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(pegid)}.csv"), index=False)
    dendrite_out = peg_syns.loc[(peg_syns.pre_root_id_720575940 == pegid) & (peg_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(pegid)}.csv"), index=False)
    dendrite_in = peg_syns.loc[(peg_syns.post_root_id_720575940 == pegid) & (peg_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(pegid)}.csv"), index=False)

# ---- PFL, PFR & PFG ----
pflr_shortids = meta.loc[meta.cell_type.str.contains('PFL|PFR|PFGs'), "root_id_short"].values
pflr_syns = syn[syn.pre_root_id_720575940.isin(pflr_shortids) | syn.post_root_id_720575940.isin(pflr_shortids)]

pflr_syns.loc[:,['compartment']] = pflr_syns.neuropil.apply(
    lambda x: "dendrite" if x in ['FB', 'PB'] else "axon"
)

for pflrid in tqdm(pflr_shortids):
    axon_out = pflr_syns.loc[(pflr_syns.pre_root_id_720575940 == pflrid) & (pflr_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(pflrid)}.csv"), index=False)
    axon_in = pflr_syns.loc[(pflr_syns.post_root_id_720575940 == pflrid) & (pflr_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(pflrid)}.csv"), index=False)
    dendrite_out = pflr_syns.loc[(pflr_syns.pre_root_id_720575940 == pflrid) & (pflr_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(pflrid)}.csv"), index=False)
    dendrite_in = pflr_syns.loc[(pflr_syns.post_root_id_720575940 == pflrid) & (pflr_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(pflrid)}.csv"), index=False)

# ---- MBONs ---- 
# only the typical MBONs 
# MBON10, MBON20, and MBON24+ are atypical MBONs according to Li et al. 2020 https://elifesciences.org/articles/62576 
# MBON21-MBON23 & MBON02, MBON04, MBON05, MBON07, MBON11-MBON19 are probably better split automatically than by neuropil, so also excluded here 
# and actually MBON06 does not split well with either neuropil or autoamtically. Will leave to automatic, but hopefully revisit later 
mbon_shortids = meta.loc[meta.cell_type.isin(['MBON01', 'MBON03', 'MBON09']), "root_id_short"].values
mbon_syns = syn[syn.pre_root_id_720575940.isin(mbon_shortids) | syn.post_root_id_720575940.isin(mbon_shortids)]

# dendrite if connection with KC, or if in the MB 
mask = (
    mbon_syns['neuropil'].str.contains('MB', na=False) |
    mbon_syns['pre_root_id_720575940'].isin(kc_shortids) |
    mbon_syns['post_root_id_720575940'].isin(kc_shortids)
)
mbon_syns.loc[:, ['compartment']] = np.where(mask, 'dendrite', 'axon')

for mbonid in tqdm(mbon_shortids):
    axon_out = mbon_syns.loc[(mbon_syns.pre_root_id_720575940 == mbonid) & (mbon_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(mbonid)}.csv"), index=False)
    axon_in = mbon_syns.loc[(mbon_syns.post_root_id_720575940 == mbonid) & (mbon_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(mbonid)}.csv"), index=False)
    dendrite_out = mbon_syns.loc[(mbon_syns.pre_root_id_720575940 == mbonid) & (mbon_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(mbonid)}.csv"), index=False)
    dendrite_in = mbon_syns.loc[(mbon_syns.post_root_id_720575940 == mbonid) & (mbon_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(mbonid)}.csv"), index=False)

# ---- DANs ---- 
# polarity of PPL2 is unclear, so leave to the automatic split 
dan_shortids = meta.loc[(meta.cell_class == 'DAN') & ~meta.cell_type.str.contains('PPL2'), "root_id_short"].values
dan_syns = syn[syn.pre_root_id_720575940.isin(dan_shortids) | syn.post_root_id_720575940.isin(dan_shortids)]

# axon if connection with KC, or if in the MB 
mask = (
    dan_syns['neuropil'].str.contains('MB', na=False) |
    dan_syns['pre_root_id_720575940'].isin(kc_shortids) |
    dan_syns['post_root_id_720575940'].isin(kc_shortids)
)
dan_syns.loc[:, ['compartment']] = np.where(mask, 'axon', 'dendrite')

for danid in tqdm(dan_shortids):
    axon_out = dan_syns.loc[(dan_syns.pre_root_id_720575940 == danid) & (dan_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(danid)}.csv"), index=False)
    axon_in = dan_syns.loc[(dan_syns.post_root_id_720575940 == danid) & (dan_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(danid)}.csv"), index=False)
    dendrite_out = dan_syns.loc[(dan_syns.pre_root_id_720575940 == danid) & (dan_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(danid)}.csv"), index=False)
    dendrite_in = dan_syns.loc[(dan_syns.post_root_id_720575940 == danid) & (dan_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(danid)}.csv"), index=False)

# ---- sensory, ascending, motor neurons ---- 
# sens_asc 
sens_asc = meta[meta.super_class.isin(['sensory','ascending','sensory_ascending'])].root_id_short.values
sens_asc_syns = syn[syn.pre_root_id_720575940.isin(sens_asc) | syn.post_root_id_720575940.isin(sens_asc)]
sens_asc_syns.loc[:,['compartment']] = 'axon'
for sid in tqdm(sens_asc):
    axon_out = sens_asc_syns.loc[(sens_asc_syns.pre_root_id_720575940 == sid)]
    axon_out = axon_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{prefix}{str(sid)}.csv"), index=False)
    axon_in = sens_asc_syns.loc[(sens_asc_syns.post_root_id_720575940 == sid)]
    axon_in = axon_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{prefix}{str(sid)}.csv"), index=False)

    
# motor 
motor = meta[meta.super_class.isin(['motor']) & (meta.cell_type != 'CB0769')].root_id_short.values
motor_syns = syn[syn.pre_root_id_720575940.isin(motor) | syn.post_root_id_720575940.isin(motor)]
motor_syns.loc[:,['compartment']] = 'dendrite'
for mid in tqdm(motor):
    dendrite_out = motor_syns.loc[(motor_syns.pre_root_id_720575940 == mid)]
    dendrite_out = dendrite_out.rename(
        columns={"pre_x": "x", "pre_y": "y", "pre_z": "z", "post_x": "partner_x", 
                 "post_y": "partner_y", "post_z": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{prefix}{str(mid)}.csv"), index=False)
    dendrite_in = motor_syns.loc[(motor_syns.post_root_id_720575940 == mid)]
    dendrite_in = dendrite_in.rename(
        columns={"post_x": "x", "post_y": "y", "post_z": "z", "pre_x": "partner_x", 
                 "pre_y": "partner_y", "pre_z": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'pre_root_id_720575940', 'post_root_id_720575940', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{prefix}{str(mid)}.csv"), index=False)

# ---- segregation index ---- 
seg_idx = []
for dir in tqdm(os.listdir(os.path.join(folder, "seg_indices"))): 
    seg_idx.append(pd.read_csv(os.path.join(folder, "seg_indices", dir), index_col=0))

seg_idx = pd.concat(seg_idx, axis=0).reset_index()
seg_idx.loc[:,['cell_type']] = seg_idx.root_id.astype(int).map(id2type)
seg_idx_type = seg_idx.groupby('cell_type').segregation_index.mean().sort_values(ascending=False).reset_index()

# ---- not split some ---- 
nosplit_types = {'APL','DPM', 'EPGt', 'Delta7', 'EL', 'ExR2_2', 'ExR2_1'}.union(set(meta.cell_type[meta.cell_type.str.contains('vDelta')]))

# add ones with low segregation index
# segregation index threshold: 0.03-0.05? 
segidx_nosplit = set(seg_idx_type[seg_idx_type.segregation_index < 0.05].cell_type.values)
nosplit_types = nosplit_types.union(segidx_nosplit)

# but remove the previously manually split ones 
splited = set(meta[meta.cell_type.str.contains('^FR') | meta.cell_type.str.contains('FS') | meta.cell_type.str.contains('FC') | 
meta.root_id_short.isin(tan_shortids) |
meta.root_id_short.isin(mbon_shortids) |
meta.root_id_short.isin(dan_shortids) |
meta.cell_sub_class.isin(['ring neuron']) | 
meta.cell_class.isin(['Kenyon_Cell']) |
meta.super_class.isin(['sensory','ascending','sensory_ascending','motor']) |
meta.cell_type.isin(['SA1', 'SA2', 'SA3',  'ExR1', 'ExR5', 'EPG']) | 
meta.cell_type.str.contains('PFN|PEN|PEG|^PFL|PFR|PFGs') 
].cell_type)

nosplit_types = nosplit_types - splited

# remove files in folders 
for folder_name in ["axon_in", "axon_out", "dendrite_in", "dendrite_out"]:
    dirs = os.listdir(os.path.join(folder, folder_name))
    rid = [int(dir.split('.')[0]) for dir in dirs]
    for r in tqdm(rid):
        if id2type[r] in nosplit_types:
            os.remove(os.path.join(folder, folder_name, f"{str(r)}.csv"))  
            
nosplit_ids = meta.loc[meta.cell_type.isin(nosplit_types), 'root_id_short'].values

syn_from_nosplit = syn[syn.pre_root_id_720575940.isin(nosplit_ids)]
syn_to_nosplit = syn[syn.post_root_id_720575940.isin(nosplit_ids)]
from_nosplit_syncount = syn_from_nosplit.groupby(['pre_root_id_720575940', 'post_root_id_720575940']).size().reset_index(name='syn_count').sort_values(by='syn_count', ascending=False)
to_nosplit_syncount = syn_to_nosplit.groupby(['pre_root_id_720575940', 'post_root_id_720575940']).size().reset_index(name='syn_count').sort_values(by='syn_count', ascending=False)
from_nosplit_syncount.to_csv(os.path.join(folder, 'syn_count', 'from_nosplit_syn_count.csv'), index=False)
to_nosplit_syncount.to_csv(os.path.join(folder, 'syn_count', 'to_nosplit_syn_count.csv'), index=False)
