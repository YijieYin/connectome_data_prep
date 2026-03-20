import glob, pandas as pd
import os

# Concatenate CSV noloop_results
files = glob.glob("noloop_results_rank*.csv")
df = pd.concat(pd.read_csv(f) for f in files)
df.to_csv("noloop_results.csv", index=False)
for f in files:
    os.remove(f)
