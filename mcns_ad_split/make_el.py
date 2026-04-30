import os
import pandas as pd
import numpy as np
from pqdm.processes import pqdm

# --- Slurm sharding ---
rank = int(os.getenv("SLURM_PROCID", "0"))
world = int(os.getenv("SLURM_NTASKS", "1"))

folder = '/cephfs2/yyin/mcns_ad_split'

meta = pd.read_csv('https://raw.githubusercontent.com/YijieYin/connectome_data_prep/refs/heads/main/data/maleCNS/mcns_all_neuron_meta.csv', index_col=0)
meta = meta.rename(columns={"superclass": "super_class", 'class': 'cell_class', 'subclass': 'cell_sub_class'})
id2type = dict(zip(meta.bodyid, meta.cell_type))

inaxonout = os.listdir(os.path.join(folder, "axon_out"))
inaxonout = [int(f.split(".")[0]) for f in inaxonout]

inaxonin = os.listdir(os.path.join(folder, "axon_in"))
inaxonin = [int(f.split(".")[0]) for f in inaxonin]

indendriteout = os.listdir(os.path.join(folder, "dendrite_out"))
indendriteout = [int(f.split(".")[0]) for f in indendriteout]

indendritein = os.listdir(os.path.join(folder, "dendrite_in"))
indendritein = [int(f.split(".")[0]) for f in indendritein]

from_nosplit_syncount = pd.read_csv(os.path.join(folder, 'syn_count', 'from_nosplit_syn_count.csv'))
to_nosplit_syncount = pd.read_csv(os.path.join(folder, 'syn_count', 'to_nosplit_syn_count.csv'))

# make the relevant folders if not exist
for subfolder in ["syn_count/aa", "syn_count/ad", "syn_count/da", "syn_count/dd", "syn_count/ba", "syn_count/bd", "syn_count/ab", "syn_count/db", "syn_count/bb"]:
    os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

def count_syn(short_id):
    ad = []
    aa = []
    da = []
    dd = []
    ba = []
    bd = []
    ab = []
    db = []
    bb = []

    # from axon 
    if short_id in inaxonout:
        axon_out = pd.read_csv(os.path.join(folder, "axon_out", f"{str(short_id)}.csv"))
        for post in set(axon_out.body_post): 
            # to axon 
            if post in inaxonin:
                a_in = pd.read_csv(os.path.join(folder, "axon_in", f"{str(post)}.csv"))
                if len(a_in) != 0: 
                    merged = axon_out.merge(a_in, 
                                            left_on = ['body_pre','body_post','x','y','z', 'partner_x','partner_y','partner_z'], 
                                            right_on = ['body_pre','body_post','partner_x','partner_y','partner_z', 'x','y','z'],
                                            suffixes = ('_pre', '_post'))
                    if len(merged) != 0:
                        aa.append(pd.DataFrame({
                            'pre_root_id': [int(f"{str(short_id)}")],
                            'post_root_id': [int(f"{str(post)}")], 
                            'syn_count': len(merged),
                        }))
            # to dendrite
            if post in indendritein:
                d_in = pd.read_csv(os.path.join(folder, "dendrite_in", f"{str(post)}.csv"))
                if len(d_in) != 0:
                    merged = axon_out.merge(d_in,
                    left_on = ['body_pre','body_post','x','y','z', 'partner_x','partner_y','partner_z'], 
                    right_on = ['body_pre','body_post','partner_x','partner_y','partner_z', 'x','y','z'],
                    suffixes = ('_pre', '_post'))
                    if len(merged) != 0:
                        ad.append(pd.DataFrame({
                            'pre_root_id': [int(f"{str(short_id)}")],
                            'post_root_id': [int(f"{str(post)}")], 
                            'syn_count': len(merged),
                        }))
            # to unsplited neuron, see later 
    
    # from dendrite
    if short_id in indendriteout:
        dendrite_out = pd.read_csv(os.path.join(folder, "dendrite_out", f"{str(short_id)}.csv"))
        for post in set(dendrite_out.body_post):
            # to axon 
            if post in inaxonin:
                a_in = pd.read_csv(os.path.join(folder, "axon_in", f"{str(post)}.csv"))
                if len(a_in) != 0: 
                    merged = dendrite_out.merge(a_in,
                    left_on = ['body_pre','body_post','x','y','z', 'partner_x','partner_y','partner_z'],
                    right_on = ['body_pre','body_post','partner_x','partner_y','partner_z', 'x','y','z'],
                    suffixes = ('_pre', '_post'))
                    if len(merged) != 0:
                        da.append(pd.DataFrame({
                            'pre_root_id': [int(f"{str(short_id)}")],
                            'post_root_id': [int(f"{str(post)}")], 
                            'syn_count': len(merged),
                        }))
            # to dendrite
            if post in indendritein:
                d_in = pd.read_csv(os.path.join(folder, "dendrite_in", f"{str(post)}.csv"))
                if len(d_in) != 0:
                    merged = dendrite_out.merge(d_in,
                    left_on = ['body_pre','body_post','x','y','z', 'partner_x','partner_y','partner_z'],
                    right_on = ['body_pre','body_post','partner_x','partner_y','partner_z', 'x','y','z'],
                    suffixes = ('_pre', '_post'))
                    if len(merged) != 0:
                        dd.append(pd.DataFrame({
                            'pre_root_id': [int(f"{str(short_id)}")],
                            'post_root_id': [int(f"{str(post)}")], 
                            'syn_count': len(merged),
                        }))
    
    if short_id in from_nosplit_syncount.body_pre.values:
        nosplit_syn = from_nosplit_syncount[from_nosplit_syncount.body_pre == short_id]
        for _, row in nosplit_syn.iterrows():
            post = row.body_post
            # if post receiving on axon 
            if post in inaxonin:
                a_in = pd.read_csv(os.path.join(folder, "axon_in", f"{str(post)}.csv"))
                if len(a_in) != 0: 
                    ba.append(pd.DataFrame({
                        'pre_root_id': [int(f"{str(short_id)}")],
                        'post_root_id': [int(f"{str(post)}")], 
                        'syn_count': min(row.syn_count, len(a_in[a_in.body_pre == short_id])),
                    }))
            # if post receiving on dendrite
            if post in indendritein:
                d_in = pd.read_csv(os.path.join(folder, "dendrite_in", f"{str(post)}.csv"))
                if len(d_in) != 0:
                    bd.append(pd.DataFrame({
                        'pre_root_id': [int(f"{str(short_id)}")],
                        'post_root_id': [int(f"{str(post)}")], 
                        'syn_count': min(row.syn_count, len(d_in[d_in.body_pre == short_id])),
                    }))
            # if post isn't divided into axon or dendrite
            if post not in inaxonin and post not in indendritein:
                bb.append(pd.DataFrame({
                    'pre_root_id': [int(f"{str(short_id)}")],
                    'post_root_id': [int(f"{str(post)}")], 
                    'syn_count': row.syn_count,
                }))
    
    if short_id in to_nosplit_syncount.body_post.values:
        nosplit_syn = to_nosplit_syncount[to_nosplit_syncount.body_post == short_id]
        for _, row in nosplit_syn.iterrows():
            pre = row.body_pre
            # pre sending from axon 
            if pre in inaxonout:
                a_out = pd.read_csv(os.path.join(folder, "axon_out", f"{str(pre)}.csv"))
                if len(a_out) != 0: 
                    ab.append(pd.DataFrame({
                        'pre_root_id': [int(f"{str(pre)}")],
                        'post_root_id': [int(f"{str(short_id)}")], 
                        'syn_count': min(row.syn_count, len(a_out[a_out.body_post == short_id])),
                    }))
            # pre sending from dendrite
            if pre in indendriteout:
                d_out = pd.read_csv(os.path.join(folder, "dendrite_out", f"{str(pre)}.csv"))
                if len(d_out) != 0:
                    db.append(pd.DataFrame({
                        'pre_root_id': [int(f"{str(pre)}")],
                        'post_root_id': [int(f"{str(short_id)}")], 
                        'syn_count': min(row.syn_count, len(d_out[d_out.body_post == short_id])),
                    }))
            # for when pre isn't divided into axon or dendrite, already taken into account above. 
    
    if len(aa) != 0: 
        aa = pd.concat(aa, axis=0).reset_index(drop=True) 
        aa = aa[(aa.syn_count > 0) & (aa.pre_root_id != aa.post_root_id)]
        aa.to_csv(os.path.join(folder, "syn_count/aa", f"{str(short_id)}.csv"), index=False)
    if len(ad) != 0: 
        ad = pd.concat(ad, axis=0).reset_index(drop=True)
        ad = ad[(ad.syn_count > 0) & (ad.pre_root_id != ad.post_root_id)]
        ad.to_csv(os.path.join(folder, "syn_count/ad", f"{str(short_id)}.csv"), index=False)
    if len(da) != 0: 
        da = pd.concat(da, axis=0).reset_index(drop=True)
        da = da[(da.syn_count > 0) & (da.pre_root_id != da.post_root_id)]
        da.to_csv(os.path.join(folder, "syn_count/da", f"{str(short_id)}.csv"), index=False)
    if len(dd) != 0: 
        dd = pd.concat(dd, axis=0).reset_index(drop=True)
        dd = dd[(dd.syn_count > 0) & (dd.pre_root_id != dd.post_root_id)]
        dd.to_csv(os.path.join(folder, "syn_count/dd", f"{str(short_id)}.csv"), index=False)
    if len(ba) != 0: 
        ba = pd.concat(ba, axis=0).reset_index(drop=True)
        ba = ba[(ba.syn_count > 0) & (ba.pre_root_id != ba.post_root_id)]
        ba.to_csv(os.path.join(folder, "syn_count/ba", f"{str(short_id)}.csv"), index=False)
    if len(bd) != 0: 
        bd = pd.concat(bd, axis=0).reset_index(drop=True)
        bd = bd[(bd.syn_count > 0) & (bd.pre_root_id != bd.post_root_id)]
        bd.to_csv(os.path.join(folder, "syn_count/bd", f"{str(short_id)}.csv"), index=False)
    if len(ab) != 0: 
        ab = pd.concat(ab, axis=0).reset_index(drop=True)
        ab = ab[(ab.syn_count > 0) & (ab.pre_root_id != ab.post_root_id)]
        ab.to_csv(os.path.join(folder, "syn_count/ab", f"{str(short_id)}.csv"), index=False)
    if len(db) != 0: 
        db = pd.concat(db, axis=0).reset_index(drop=True)
        db = db[(db.syn_count > 0) & (db.pre_root_id != db.post_root_id)]
        db.to_csv(os.path.join(folder, "syn_count/db", f"{str(short_id)}.csv"), index=False)
    if len(bb) != 0: 
        bb = pd.concat(bb, axis=0).reset_index(drop=True)
        bb = bb[(bb.syn_count > 0) & (bb.pre_root_id != bb.post_root_id)]
        bb.to_csv(os.path.join(folder, "syn_count/bb", f"{str(short_id)}.csv"), index=False)


if __name__ == "__main__":
    all_ids = meta.bodyid.values
    my_ids = np.array_split(all_ids, world)[rank]
    
    pqdm(my_ids, count_syn, n_jobs=112)