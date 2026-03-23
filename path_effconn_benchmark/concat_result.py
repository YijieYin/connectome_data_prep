import glob, pandas as pd
import os

# Concatenate CSV custom_effconn_results
files = glob.glob("custom_effconn_results_rank*.csv")
df = pd.concat(pd.read_csv(f) for f in files)
df.to_csv("custom_effconn_results.csv", index=False)
for f in files:
    os.remove(f)
