"""Sanity-check the finished a/d split.

Run after make_adj.py. Prints a report and exits non-zero if any check fails, so
`set -e` in ad_split.sh turns a silently-wrong result into a failed job.

The headline check is conservation: every synapse in the (de-duplicated) synapse
table whose partners are both in meta must appear exactly once across the nine
edge lists. That catches both the neurons that drop out of the pipeline and the
ones that get counted twice.
"""
import os
import sys

import numpy as np
import pandas as pd

prefix = "720575940"
folder = '/cephfs2/yyin/ad_split'

# (pre compartment, post compartment) -> which edge lists carry it. 'b' = not split
BRANCHES = {
    ('split', 'split'): ['aa', 'ad', 'da', 'dd'],
    ('split', 'nosplit'): ['ab', 'db'],
    ('nosplit', 'split'): ['ba', 'bd'],
    ('nosplit', 'nosplit'): ['bb'],
}
CLASSES = ['split', 'nosplit', 'unclassified']

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


meta = pd.read_csv('https://raw.githubusercontent.com/YijieYin/connectome_data_prep/'
                   'refs/heads/main/data/fafb_all_neuron/fafb_all_neuron_meta.csv', index_col=0)
meta.loc[:, ["root_id_short"]] = meta.root_id.apply(lambda x: int(str(x).split(prefix)[1]))
selected = set(meta.root_id_short[
    (meta.super_class.isin(["optic", "central", "visual_projection", "visual_centrifugal",
                            "descending", "endocrine"])
     & (meta.cell_class != 'Kenyon_Cell') & (~meta.cell_type.isin(['APL', 'DPM'])))
    | (meta.cell_type == 'CB0769')
])

# ---- who ended up split, and who was left whole ----
split_ids = set()
for sub in ["axon_in", "axon_out", "dendrite_in", "dendrite_out"]:
    split_ids |= set(int(f[len(prefix):-4]) for f in os.listdir(os.path.join(folder, sub)))
nosplit_ids = set()
for f, col in [('from_nosplit_syn_count.csv', 'pre_root_id_720575940'),
               ('to_nosplit_syn_count.csv', 'post_root_id_720575940')]:
    nosplit_ids |= set(pd.read_csv(os.path.join(folder, 'syn_count', f), usecols=[col])[col].astype('int64').unique())

split_arr = np.fromiter(split_ids, dtype=np.int64)
nosplit_arr = np.fromiter(nosplit_ids, dtype=np.int64)

# ---- one pass over the synapse table: de-duplicate, then classify ----
# the pipeline drops ctr_*/size and de-duplicates, so do the same here
COLS = ['pre_x', 'pre_y', 'pre_z', 'post_x', 'post_y', 'post_z',
        'pre_root_id_720575940', 'post_root_id_720575940', 'neuropil']
npil = {}
hashes, codes = [], []
seen_pre, seen_post = set(), set()
for chunk in pd.read_csv(os.path.join(folder, "fafb_v783_princeton_synapse_table.csv"),
                         usecols=COLS, chunksize=2_000_000):
    chunk = chunk[chunk.pre_root_id_720575940 != chunk.post_root_id_720575940]
    seen_pre |= set(chunk.pre_root_id_720575940.unique())
    seen_post |= set(chunk.post_root_id_720575940.unique())
    for name in chunk.neuropil.unique():
        npil.setdefault(name, len(npil))
    h = np.zeros(len(chunk), dtype=np.uint64)
    for c in COLS[:-1]:
        h = h * np.uint64(1000003) + chunk[c].values.astype(np.int64).view(np.uint64)
    h = h * np.uint64(1000003) + chunk.neuropil.map(npil).values.astype(np.int64).view(np.uint64)
    hashes.append(h)
    pre = np.where(np.isin(chunk.pre_root_id_720575940.values, split_arr), 0,
                   np.where(np.isin(chunk.pre_root_id_720575940.values, nosplit_arr), 1, 2))
    post = np.where(np.isin(chunk.post_root_id_720575940.values, split_arr), 0,
                    np.where(np.isin(chunk.post_root_id_720575940.values, nosplit_arr), 1, 2))
    codes.append((pre * 3 + post).astype(np.uint8))
hashes = np.concatenate(hashes)
codes = np.concatenate(codes)
_, keep = np.unique(hashes, return_index=True)
print(f"synapse table: {len(hashes)} rows without autapses, {len(keep)} after de-duplication")
counts = np.bincount(codes[keep], minlength=9)
truth = {(CLASSES[i // 3], CLASSES[i % 3]): int(counts[i]) for i in range(9)}
del hashes, codes

# make_el sends a nosplit pre onto any post that isn't split into bb, which includes
# posts with no meta entry - count those as bb rather than as excluded
no_meta_in_bb = truth[('nosplit', 'unclassified')]
truth[('nosplit', 'nosplit')] += no_meta_in_bb
truth[('nosplit', 'unclassified')] = 0

# ---- what the edge lists actually contain ----
observed = {}
all_pre, all_post = set(), set()
for key, names in BRANCHES.items():
    observed[key] = 0
    for n in names:
        el = pd.read_csv(os.path.join(folder, "result", f"{n}_el.csv"))
        observed[key] += el.syn_count.sum()
        all_pre |= set(el.pre_root_id.astype(str).str[len(prefix):].astype('int64').unique())
        all_post |= set(el.post_root_id.astype(str).str[len(prefix):].astype('int64').unique())
        if (el.syn_count <= 0).any():
            failures.append(f"{n}_el.csv has non-positive syn_count")
        if (el.pre_root_id == el.post_root_id).any():
            failures.append(f"{n}_el.csv contains self-edges")
total_observed = sum(observed.values())
total_truth = sum(truth[k] for k in BRANCHES)

print(f"\n{'':<24}{'edge lists':>14}{'synapse table':>16}{'diff':>10}")
for key in BRANCHES:
    print(f"{key[0] + ' -> ' + key[1]:<24}{observed[key]:>14}{truth[key]:>16}{observed[key] - truth[key]:>+10}")
print(f"{'TOTAL':<24}{total_observed:>14}{total_truth:>16}{total_observed - total_truth:>+10}")
outside = sum(v for k, v in truth.items() if 'unclassified' in k)
print(f"\n(a further {outside} synapses involve a neuron that is neither split nor left "
      f"whole - these have no meta entry and are excluded by design; a separate "
      f"{no_meta_in_bb} such synapses are counted in bb, see above)\n")

check("every synapse counted exactly once", total_observed == total_truth,
      f"{total_observed - total_truth:+d} of {total_truth}")
for key in BRANCHES:
    check(f"{key[0]} -> {key[1]} conserved", observed[key] == truth[key],
          f"{observed[key] - truth[key]:+d}")

# ---- the split stage ----
fail_dir = os.path.join(folder, "split_failures")
failed = []
if os.path.isdir(fail_dir):
    for f in os.listdir(fail_dir):
        failed.append(pd.read_csv(os.path.join(fail_dir, f)))
failed = pd.concat(failed) if failed else pd.DataFrame(columns=['root_id_short', 'error'])
check("no neuron errored during the split", len(failed) == 0, f"{len(failed)} failed")
if len(failed):
    print(failed.error.value_counts().head(5).to_string())

n_seg = len(os.listdir(os.path.join(folder, "seg_indices")))
one_polarity = seen_pre.symmetric_difference(seen_post) | (selected - seen_pre - seen_post)
skipped = len(selected & one_polarity)
check("every selected neuron has a segregation index or was skipped",
      n_seg + skipped + len(failed) == len(selected),
      f"{n_seg} seg + {skipped} single-polarity + {len(failed)} failed vs {len(selected)} selected")

# ---- partitioning ----
check("no neuron is both split and left whole", len(split_ids & nosplit_ids) == 0,
      f"{len(split_ids & nosplit_ids)} overlap")
check("every selected neuron is split or left whole", len(selected - split_ids - nosplit_ids) == 0,
      f"{len(selected - split_ids - nosplit_ids)} in neither")
check("every selected neuron appears in the results", len(selected - all_pre - all_post) == 0,
      f"{len(selected - all_pre - all_post)} absent")

# ---- stale compartment files (a manual rule that writes only 2 of the 4) ----
KEY = ['pre_root_id_720575940', 'post_root_id_720575940', 'x', 'y', 'z',
       'partner_x', 'partner_y', 'partner_z']
at_risk = set(meta.root_id_short[
    meta.super_class.isin(['sensory', 'ascending', 'sensory_ascending', 'motor'])
    | (meta.cell_type == 'IPC')]) & split_ids
dup = 0
for i in at_risk:
    for a, b in [("axon_in", "dendrite_in"), ("axon_out", "dendrite_out")]:
        pa_, pb = os.path.join(folder, a, f"{prefix}{i}.csv"), os.path.join(folder, b, f"{prefix}{i}.csv")
        if not (os.path.exists(pa_) and os.path.exists(pb)):
            continue
        x, y = pd.read_csv(pa_), pd.read_csv(pb)
        if len(x) and len(y):
            dup += len(set(map(tuple, x[KEY].values)) & set(map(tuple, y[KEY].values)))
check("no synapse sits in two compartments of the same neuron", dup == 0,
      f"{dup} duplicated across {len(at_risk)} neurons at risk")

print(f"\nfor information: {len(all_pre)} neurons appear as pre, {len(all_post)} as post "
      f"({len(all_post - all_pre)} post-only, {len(all_pre - all_post)} pre-only)")

if failures:
    print(f"\n{len(failures)} CHECK(S) FAILED: " + "; ".join(failures))
    sys.exit(1)
print("\nall checks passed")
