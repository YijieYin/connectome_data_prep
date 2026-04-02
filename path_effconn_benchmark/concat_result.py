import glob, pandas as pd
import os

# Concatenate CSV ad_sensory_dn_thresholds_onesyn
files = glob.glob("ad_sensory_dn_thresholds_onesyn_rank*.csv")
df = pd.concat(pd.read_csv(f) for f in files)
df.to_csv("ad_sensory_dn_thresholds_onesyn.csv", index=False)
for f in files:
    os.remove(f)
