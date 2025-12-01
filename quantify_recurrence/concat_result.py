import glob, pandas as pd
import os

files = glob.glob("recurrence_results_rank*.csv")
df = pd.concat(pd.read_csv(f) for f in files)
df.to_csv("recurrence_results.csv", index=False)
for f in files:
    os.remove(f)
