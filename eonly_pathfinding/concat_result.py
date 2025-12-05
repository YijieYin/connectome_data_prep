import glob, pandas as pd
import os

files = glob.glob("ionly_pathfinding_results_rank*.csv")
df = pd.concat(pd.read_csv(f) for f in files)
df.to_csv("ionly_pathfinding_results.csv", index=False)
for f in files:
    os.remove(f)
