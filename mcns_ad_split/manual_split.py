import os 
import pandas as pd
import numpy as np
from tqdm import tqdm

folder = '/cephfs2/yyin/mcns_ad_split'

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

# make the relevant folders if not exist
for subfolder in ["syn_count"]:
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

# ---- KC ---- 
kc_shortids = meta.loc[meta.cell_class == "Kenyon_Cell", "bodyid"].values
kc_syns = syn[syn.body_pre.isin(kc_shortids) | syn.body_post.isin(kc_shortids)]
kc_syns.loc[:,['compartment']] = kc_syns.neuropil.apply(
    lambda x: "dendrite" if x in ["CA(L)", 'CA(R)', "LH(L)", 'LH(R)', "PLP(L)", 'PLP(R)',
                                  "SCL(L)", 'SCL(R)', "ICL(L)", "ICL(R)", "SLP(L)", 'SLP(R)', 
                                  "PB", "LO(L)", 'LO(R)', "ATL(L)", 'ATL(R)'] else "axon"
)
for kcid in tqdm(meta.bodyid[meta.cell_class == "Kenyon_Cell"].values):
    axon_out = kc_syns.loc[(kc_syns.body_pre == kcid) & (kc_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(kcid)}.csv"), index=False)
    axon_in = kc_syns.loc[(kc_syns.body_post == kcid) & (kc_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(kcid)}.csv"), index=False)
    dendrite_out = kc_syns.loc[(kc_syns.body_pre == kcid) & (kc_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(kcid)}.csv"), index=False)
    dendrite_in = kc_syns.loc[(kc_syns.body_post == kcid) & (kc_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(kcid)}.csv"), index=False)


# ---- tangential neurons ---- 
tan_shortids = meta.loc[(meta.cell_type.str.contains('^FB')) 
& (meta.super_class == 'cb_intrinsic') 
# FB intrinsic types, not spliting 
& ~meta.cell_type.str.contains('^FB4Z|^FB5R|^FB5U|^FB5S|^FB6J|^FB6L|^FB7D|^FB7J'), 
"bodyid"].values
tan_syns = syn[syn.body_pre.isin(tan_shortids) | syn.body_post.isin(tan_shortids)]

tan_syns.loc[:,['compartment']] = tan_syns.neuropil.apply(
    lambda x: "axon" if x in ["FB",'NO'] else "dendrite"
)

for tanid in tqdm(tan_shortids):
    axon_out = tan_syns.loc[(tan_syns.body_pre == tanid) & (tan_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(tanid)}.csv"), index=False)
    axon_in = tan_syns.loc[(tan_syns.body_post == tanid) & (tan_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(tanid)}.csv"), index=False)
    dendrite_out = tan_syns.loc[(tan_syns.body_pre == tanid) & (tan_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(tanid)}.csv"), index=False)
    dendrite_in = tan_syns.loc[(tan_syns.body_post == tanid) & (tan_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(tanid)}.csv"), index=False)


# ---- FR, FS, FC ---- 
frsc_shortids = meta.loc[meta.cell_type.str.contains('^FR') | meta.cell_type.str.contains('FS') | meta.cell_type.str.contains('FC'), "bodyid"].values
frsc_syns = syn[syn.body_pre.isin(frsc_shortids) | syn.body_post.isin(frsc_shortids)]

frsc_syns.loc[:,['compartment']] = frsc_syns.neuropil.apply(
    lambda x: "dendrite" if x in ["FB"] else "axon"
)
for frscid in tqdm(frsc_shortids):
    axon_out = frsc_syns.loc[(frsc_syns.body_pre == frscid) & (frsc_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(frscid)}.csv"), index=False)
    axon_in = frsc_syns.loc[(frsc_syns.body_post == frscid) & (frsc_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(frscid)}.csv"), index=False)
    dendrite_out = frsc_syns.loc[(frsc_syns.body_pre == frscid) & (frsc_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(frscid)}.csv"), index=False)
    dendrite_in = frsc_syns.loc[(frsc_syns.body_post == frscid) & (frsc_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(frscid)}.csv"), index=False)


# ---- SA1-3 ---- 
sa123_shortids = meta.loc[meta.cell_type.isin(['SA1', 'SA2', 'SA3']), 'bodyid'].values
sa123_syns = syn[syn.body_pre.isin(sa123_shortids) | syn.body_post.isin(sa123_shortids)]
sa123_syns.loc[:,['compartment']] = sa123_syns.neuropil.apply(
    lambda x: "axon" if x in ["FB", 'NO'] else "dendrite"
)
for sa123id in tqdm(sa123_shortids):
    axon_out = sa123_syns.loc[(sa123_syns.body_pre == sa123id) & (sa123_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(sa123id)}.csv"), index=False)
    axon_in = sa123_syns.loc[(sa123_syns.body_post == sa123id) & (sa123_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(sa123id)}.csv"), index=False)
    dendrite_out = sa123_syns.loc[(sa123_syns.body_pre == sa123id) & (sa123_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(sa123id)}.csv"), index=False)
    dendrite_in = sa123_syns.loc[(sa123_syns.body_post == sa123id) & (sa123_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(sa123id)}.csv"), index=False)

# ---- ExR5 ---- 
exr5_shortids = meta.loc[meta.cell_type == 'ExR5', "bodyid"].values
exr5_syns = syn[syn.body_pre.isin(exr5_shortids) | syn.body_post.isin(exr5_shortids)]
exr5_syns.loc[:,['compartment']] = exr5_syns.neuropil.apply(
    lambda x: "dendrite" if x in ["SPS(L)", 'SPS(R)', 'IB', 'ICL(L)', 'ICL(R)', 'PLP(R)', 'PLP(L)', 'IPS(R)', 'IPS(L)', 'ATL(R)', 'ATL(L)', 'PB'] else "axon"
)
for exr5id in tqdm(exr5_shortids):
    axon_out = exr5_syns.loc[(exr5_syns.body_pre == exr5id) & (exr5_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(exr5id)}.csv"), index=False)
    axon_in = exr5_syns.loc[(exr5_syns.body_post == exr5id) & (exr5_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(exr5id)}.csv"), index=False)
    dendrite_out = exr5_syns.loc[(exr5_syns.body_pre == exr5id) & (exr5_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(exr5id)}.csv"), index=False)
    dendrite_in = exr5_syns.loc[(exr5_syns.body_post == exr5id) & (exr5_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(exr5id)}.csv"), index=False)

# ---- ExR1 ----
exr1_shortids = meta.loc[meta.cell_type == 'ExR1', "bodyid"].values
exr1_syns = syn[syn.body_pre.isin(exr1_shortids) | syn.body_post.isin(exr1_shortids)]

exr1_syns.loc[:,['compartment']] = exr1_syns.neuropil.apply(
    lambda x: "axon" if x in ["EB"] else "dendrite"
)
for exr1id in tqdm(exr1_shortids):
    axon_out = exr1_syns.loc[(exr1_syns.body_pre == exr1id) & (exr1_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(exr1id)}.csv"), index=False)
    axon_in = exr1_syns.loc[(exr1_syns.body_post == exr1id) & (exr1_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(exr1id)}.csv"), index=False)
    dendrite_out = exr1_syns.loc[(exr1_syns.body_pre == exr1id) & (exr1_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(exr1id)}.csv"), index=False)
    dendrite_in = exr1_syns.loc[(exr1_syns.body_post == exr1id) & (exr1_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(exr1id)}.csv"), index=False)

# ---- ring neurons ---- 
ring_shortids = meta.loc[meta.cell_type.str.contains('^ER'), "bodyid"].values
ring_syns = syn[syn.body_pre.isin(ring_shortids) | syn.body_post.isin(ring_shortids)]
ring_syns.loc[:,['compartment']] = ring_syns.neuropil.apply(
    lambda x: "axon" if x in ['EB'] else "dendrite"
)
for ringid in tqdm(ring_shortids):
    axon_out = ring_syns.loc[(ring_syns.body_pre == ringid) & (ring_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(ringid)}.csv"), index=False)
    axon_in = ring_syns.loc[(ring_syns.body_post == ringid) & (ring_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(ringid)}.csv"), index=False)
    dendrite_out = ring_syns.loc[(ring_syns.body_pre == ringid) & (ring_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(ringid)}.csv"), index=False)
    dendrite_in = ring_syns.loc[(ring_syns.body_post == ringid) & (ring_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(ringid)}.csv"), index=False)

# ---- PFN ----
pfn_shortids = meta.loc[meta.cell_type.str.contains('PFN'), "bodyid"].values
pfn_syns = syn[syn.body_pre.isin(pfn_shortids) | syn.body_post.isin(pfn_shortids)]

pfn_syns.loc[:,['compartment']] = pfn_syns.neuropil.apply(
    lambda x: "axon" if x in ['FB'] else "dendrite"
)

for pfnid in tqdm(pfn_shortids):
    axon_out = pfn_syns.loc[(pfn_syns.body_pre == pfnid) & (pfn_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(pfnid)}.csv"), index=False)
    axon_in = pfn_syns.loc[(pfn_syns.body_post == pfnid) & (pfn_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(pfnid)}.csv"), index=False)
    dendrite_out = pfn_syns.loc[(pfn_syns.body_pre == pfnid) & (pfn_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(pfnid)}.csv"), index=False)
    dendrite_in = pfn_syns.loc[(pfn_syns.body_post == pfnid) & (pfn_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(pfnid)}.csv"), index=False)

# ---- PEN ---- 
pen_shortids = meta.loc[meta.cell_type.str.contains('PEN'), "bodyid"].values
pen_syns = syn[syn.body_pre.isin(pen_shortids) | syn.body_post.isin(pen_shortids)]

pen_syns.loc[:,['compartment']] = pen_syns.neuropil.apply(
    lambda x: "axon" if x in ['EB'] else "dendrite"
)

for penid in tqdm(pen_shortids):
    axon_out = pen_syns.loc[(pen_syns.body_pre == penid) & (pen_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(penid)}.csv"), index=False)
    axon_in = pen_syns.loc[(pen_syns.body_post == penid) & (pen_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(penid)}.csv"), index=False)
    dendrite_out = pen_syns.loc[(pen_syns.body_pre == penid) & (pen_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(penid)}.csv"), index=False)
    dendrite_in = pen_syns.loc[(pen_syns.body_post == penid) & (pen_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(penid)}.csv"), index=False)

# ---- EPG ----
epg_shortids = meta.loc[meta.cell_type == 'EPG', "bodyid"].values
epg_syns = syn[syn.body_pre.isin(epg_shortids) | syn.body_post.isin(epg_shortids)]

epg_syns.loc[:,['compartment']] = epg_syns.neuropil.apply(
    lambda x: "dendrite" if x in ['EB'] else "axon"
)

for epgid in tqdm(epg_shortids):
    axon_out = epg_syns.loc[(epg_syns.body_pre == epgid) & (epg_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(epgid)}.csv"), index=False)
    axon_in = epg_syns.loc[(epg_syns.body_post == epgid) & (epg_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(epgid)}.csv"), index=False)
    dendrite_out = epg_syns.loc[(epg_syns.body_pre == epgid) & (epg_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(epgid)}.csv"), index=False)
    dendrite_in = epg_syns.loc[(epg_syns.body_post == epgid) & (epg_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(epgid)}.csv"), index=False)

# ---- PEG ----
peg_shortids = meta.loc[meta.cell_type.str.contains('PEG'), "bodyid"].values
peg_syns = syn[syn.body_pre.isin(peg_shortids) | syn.body_post.isin(peg_shortids)]

peg_syns.loc[:,['compartment']] = peg_syns.neuropil.apply(
    lambda x: "dendrite" if x in ['PB'] else "axon"
)

for pegid in tqdm(peg_shortids):
    axon_out = peg_syns.loc[(peg_syns.body_pre == pegid) & (peg_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(pegid)}.csv"), index=False)
    axon_in = peg_syns.loc[(peg_syns.body_post == pegid) & (peg_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(pegid)}.csv"), index=False)
    dendrite_out = peg_syns.loc[(peg_syns.body_pre == pegid) & (peg_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(pegid)}.csv"), index=False)
    dendrite_in = peg_syns.loc[(peg_syns.body_post == pegid) & (peg_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(pegid)}.csv"), index=False)

# ---- PFL, PFR & PFG ----
pflr_shortids = meta.loc[meta.cell_type.str.contains('PFL|PFR|PFGs'), "bodyid"].values
pflr_syns = syn[syn.body_pre.isin(pflr_shortids) | syn.body_post.isin(pflr_shortids)]

pflr_syns.loc[:,['compartment']] = pflr_syns.neuropil.apply(
    lambda x: "dendrite" if x in ['FB', 'PB'] else "axon"
)

for pflrid in tqdm(pflr_shortids):
    axon_out = pflr_syns.loc[(pflr_syns.body_pre == pflrid) & (pflr_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(pflrid)}.csv"), index=False)
    axon_in = pflr_syns.loc[(pflr_syns.body_post == pflrid) & (pflr_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(pflrid)}.csv"), index=False)
    dendrite_out = pflr_syns.loc[(pflr_syns.body_pre == pflrid) & (pflr_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(pflrid)}.csv"), index=False)
    dendrite_in = pflr_syns.loc[(pflr_syns.body_post == pflrid) & (pflr_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(pflrid)}.csv"), index=False)

# ---- MBONs ---- 
# only the typical MBONs 
# MBON10, MBON20, and MBON24+ are atypical MBONs according to Li et al. 2020 https://elifesciences.org/articles/62576 
# MBON21-MBON23 & MBON02, MBON04, MBON05, MBON07, MBON11-MBON19 are probably better split automatically than by neuropil, so also excluded here 
# and actually MBON06 does not split well with either neuropil or autoamtically. Will leave to automatic, but hopefully revisit later 
mbon_shortids = meta.loc[meta.cell_type.isin(['MBON01', 'MBON03', 'MBON09']), "bodyid"].values
mbon_syns = syn[syn.body_pre.isin(mbon_shortids) | syn.body_post.isin(mbon_shortids)]

# dendrite if connection with KC, or if in the MB 
mask = (
    mbon_syns['neuropil'].isin(['CA(L)','CA(R)',
    'PED(L)','PED(R)', 
    "a'L(L)","a'L(R)",'aL(L)','aL(R)',
    "b'L(L)","b'L(R)",'bL(L)','bL(R)',
    'gL(L)','gL(R)']) |
    mbon_syns['body_pre'].isin(kc_shortids) |
    mbon_syns['body_post'].isin(kc_shortids)
)
mbon_syns.loc[:, ['compartment']] = np.where(mask, 'dendrite', 'axon')

for mbonid in tqdm(mbon_shortids):
    axon_out = mbon_syns.loc[(mbon_syns.body_pre == mbonid) & (mbon_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(mbonid)}.csv"), index=False)
    axon_in = mbon_syns.loc[(mbon_syns.body_post == mbonid) & (mbon_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(mbonid)}.csv"), index=False)
    dendrite_out = mbon_syns.loc[(mbon_syns.body_pre == mbonid) & (mbon_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(mbonid)}.csv"), index=False)
    dendrite_in = mbon_syns.loc[(mbon_syns.body_post == mbonid) & (mbon_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(mbonid)}.csv"), index=False)

# ---- DANs ---- 
# polarity of PPL2 is unclear, so leave to the automatic split 
dan_shortids = meta.loc[(meta.cell_class == 'DAN') & ~meta.cell_type.str.contains('PPL2'), "bodyid"].values
dan_syns = syn[syn.body_pre.isin(dan_shortids) | syn.body_post.isin(dan_shortids)]

# axon if connection with KC, or if in the MB 
mask = (
    dan_syns['neuropil'].str.contains('MB', na=False) |
    dan_syns['body_pre'].isin(kc_shortids) |
    dan_syns['body_post'].isin(kc_shortids)
)
dan_syns.loc[:, ['compartment']] = np.where(mask, 'axon', 'dendrite')

for danid in tqdm(dan_shortids):
    axon_out = dan_syns.loc[(dan_syns.body_pre == danid) & (dan_syns.compartment == "axon")]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(danid)}.csv"), index=False)
    axon_in = dan_syns.loc[(dan_syns.body_post == danid) & (dan_syns.compartment == "axon")]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(danid)}.csv"), index=False)
    dendrite_out = dan_syns.loc[(dan_syns.body_pre == danid) & (dan_syns.compartment == "dendrite")]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(danid)}.csv"), index=False)
    dendrite_in = dan_syns.loc[(dan_syns.body_post == danid) & (dan_syns.compartment == "dendrite")]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(danid)}.csv"), index=False)

# ---- sensory, motor neurons, and IPC (dendrites only in CNS I think) ---- 
sens_asc = meta[meta.super_class.str.contains('sensory') | (meta.cell_type == 'IPC')].bodyid.values
sens_asc_syns = syn[syn.body_pre.isin(sens_asc) | syn.body_post.isin(sens_asc)]
sens_asc_syns.loc[:,['compartment']] = 'axon'
for sid in tqdm(sens_asc):
    axon_out = sens_asc_syns.loc[(sens_asc_syns.body_pre == sid)]
    axon_out = axon_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    axon_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_out", f"{str(sid)}.csv"), index=False)
    axon_in = sens_asc_syns.loc[(sens_asc_syns.body_post == sid)]
    axon_in = axon_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    axon_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "axon_in", f"{str(sid)}.csv"), index=False)
    # axon only, so drop the dendrite files axon_dendrite_split.py may have written
    # (IPC isn't excluded from selected_skids) - otherwise make_el.py counts those
    # synapses both here and there
    for compartment in ["dendrite_in", "dendrite_out"]:
        p = os.path.join(folder, compartment, f"{str(sid)}.csv")
        if os.path.exists(p):
            os.remove(p)

# motor
motor = meta[meta.super_class.str.contains('motor|efferent')].bodyid.values
motor_syns = syn[syn.body_pre.isin(motor) | syn.body_post.isin(motor)]
motor_syns.loc[:,['compartment']] = 'dendrite'
for mid in tqdm(motor):
    dendrite_out = motor_syns.loc[(motor_syns.body_pre == mid)]
    dendrite_out = dendrite_out.rename(
        columns={"x_pre": "x", "y_pre": "y", "z_pre": "z", "x_post": "partner_x", 
                 "y_post": "partner_y", "z_post": "partner_z"},)
    dendrite_out[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_out", f"{str(mid)}.csv"), index=False)
    dendrite_in = motor_syns.loc[(motor_syns.body_post == mid)]
    dendrite_in = dendrite_in.rename(
        columns={"x_post": "x", "y_post": "y", "z_post": "z", "x_pre": "partner_x", 
                 "y_pre": "partner_y", "z_pre": "partner_z"},)
    dendrite_in[['x', 'y', 'z', 'neuropil', 'body_pre', 'body_post', 'partner_x', 'partner_y', 'partner_z']].to_csv(os.path.join(folder, "dendrite_in", f"{str(mid)}.csv"), index=False)
    # dendrite only, so drop the axon files axon_dendrite_split.py may have written
    # (efferent isn't excluded from selected_skids) - otherwise make_el.py counts
    # those synapses both here and there
    for compartment in ["axon_in", "axon_out"]:
        p = os.path.join(folder, compartment, f"{str(mid)}.csv")
        if os.path.exists(p):
            os.remove(p)

# ---- segregation index ----
seg_idx = []
for dir in tqdm(os.listdir(os.path.join(folder, "seg_indices"))): 
    seg_idx.append(pd.read_csv(os.path.join(folder, "seg_indices", dir), index_col=0))

seg_idx = pd.concat(seg_idx, axis=0).reset_index()
seg_idx.loc[:,['cell_type']] = seg_idx.root_id.astype(int).map(id2type)
seg_idx_type = seg_idx.groupby('cell_type').segregation_index.mean().sort_values(ascending=False).reset_index()

# ---- not split some ---- 
nosplit_types = {'APL','DPM', 'EPGt', 'Delta7', 'EL', 'ExR2'}.union(set(meta.cell_type[meta.cell_type.str.contains('vDelta')]))

# add ones with low segregation index
# segregation index threshold: 0.03-0.05? 
segidx_nosplit = set(seg_idx_type[seg_idx_type.segregation_index < 0.05].cell_type.values)
nosplit_types = nosplit_types.union(segidx_nosplit)

# neurons with only pre- or only postsynapses cannot be split by navis, see
# axon_dendrite_split.py. Split/no-split is decided per cell type, so don't split
# the types they belong to
one_polarity = set(syn.body_pre.unique()).symmetric_difference(set(syn.body_post.unique()))
nosplit_types = nosplit_types.union(set(meta.cell_type[meta.bodyid.isin(one_polarity)]))

# but remove the previously manually split ones 
splited = set(meta[meta.cell_type.str.contains('^FR') | meta.cell_type.str.contains('FS') | meta.cell_type.str.contains('FC') | 
meta.bodyid.isin(tan_shortids) |
meta.bodyid.isin(mbon_shortids) |
meta.bodyid.isin(dan_shortids) |
meta.bodyid.isin(ring_shortids) | 
meta.cell_class.isin(['Kenyon_Cell']) |
meta.super_class.str.contains('sensory|motor|efferent') | 
meta.cell_type.isin(['SA1', 'SA2', 'SA3',  'ExR1', 'ExR5', 'EPG', 'IPC']) | 
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
            
nosplit_ids = meta.loc[meta.cell_type.isin(nosplit_types), 'bodyid'].values

syn_from_nosplit = syn[syn.body_pre.isin(nosplit_ids)]
syn_to_nosplit = syn[syn.body_post.isin(nosplit_ids)]
from_nosplit_syncount = syn_from_nosplit.groupby(['body_pre', 'body_post']).size().reset_index(name='syn_count').sort_values(by='syn_count', ascending=False)
to_nosplit_syncount = syn_to_nosplit.groupby(['body_pre', 'body_post']).size().reset_index(name='syn_count').sort_values(by='syn_count', ascending=False)
from_nosplit_syncount.to_csv(os.path.join(folder, 'syn_count', 'from_nosplit_syn_count.csv'), index=False)
to_nosplit_syncount.to_csv(os.path.join(folder, 'syn_count', 'to_nosplit_syn_count.csv'), index=False)
