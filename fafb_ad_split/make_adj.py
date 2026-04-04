import navis
import os 
import pandas as pd
from tqdm import tqdm

prefix = "720575940"
folder = '/cephfs2/yyin/ad_split'

# concatenate all 
aa = []
ad = []
da = []
dd = []
ba = []
bd = []
ab = []
db = []
bb = []


for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/aa"))):
    aa.append(pd.read_csv(os.path.join(folder, "syn_count/aa", adir)))

aa = pd.concat(aa, axis=0).reset_index(drop=True)
aa.to_csv(os.path.join(folder, "result", "aa_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/ad"))):
    ad.append(pd.read_csv(os.path.join(folder, "syn_count/ad", adir)))

ad = pd.concat(ad, axis=0).reset_index(drop=True)
ad.to_csv(os.path.join(folder, "result", "ad_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/da"))):
    da.append(pd.read_csv(os.path.join(folder, "syn_count/da", adir)))

da = pd.concat(da, axis=0).reset_index(drop=True)
da.to_csv(os.path.join(folder, "result", "da_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/dd"))):
    dd.append(pd.read_csv(os.path.join(folder, "syn_count/dd", adir)))

dd = pd.concat(dd, axis=0).reset_index(drop=True)
dd.to_csv(os.path.join(folder, "result", "dd_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/ba"))):
    ba.append(pd.read_csv(os.path.join(folder, "syn_count/ba", adir)))

ba = pd.concat(ba, axis=0).reset_index(drop=True)
ba.to_csv(os.path.join(folder, "result", "ba_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/bd"))):
    bd.append(pd.read_csv(os.path.join(folder, "syn_count/bd", adir)))

bd = pd.concat(bd, axis=0).reset_index(drop=True)
bd.to_csv(os.path.join(folder, "result", "bd_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/ab"))):
    ab.append(pd.read_csv(os.path.join(folder, "syn_count/ab", adir)))

ab = pd.concat(ab, axis=0).reset_index(drop=True)
ab.to_csv(os.path.join(folder, "result", "ab_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/db"))):
    db.append(pd.read_csv(os.path.join(folder, "syn_count/db", adir)))

db = pd.concat(db, axis=0).reset_index(drop=True)
db.to_csv(os.path.join(folder, "result", "db_el.csv"), index=False)

for adir in tqdm(os.listdir(os.path.join(folder, "syn_count/bb"))):
    bb.append(pd.read_csv(os.path.join(folder, "syn_count/bb", adir)))
    
bb = pd.concat(bb, axis=0).reset_index(drop=True)
bb.to_csv(os.path.join(folder, "result", "bb_el.csv"), index=False)

