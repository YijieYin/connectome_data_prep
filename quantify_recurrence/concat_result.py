import glob, pandas as pd
import numpy as np
import os

# Concatenate CSV results
files = glob.glob("sc_recurrence_results_rank*.csv")
df = pd.concat(pd.read_csv(f) for f in files)
df.to_csv("sc_recurrence_results.csv", index=False)
for f in files:
    os.remove(f)
