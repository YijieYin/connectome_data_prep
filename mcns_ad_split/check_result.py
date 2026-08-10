"""Sanity-check the finished a/d split.

Run after make_adj.py. Prints a report and exits non-zero if any check fails, so
`set -e` in ad_split.sh turns a silently-wrong result into a failed job.

The headline check is conservation: every synapse in the (filtered) synapse table
must appear exactly once across the nine edge lists. That catches both the
neurons that drop out of the pipeline and the ones that get counted twice.
"""
import os
import sys
import collections

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

folder = '/cephfs2/yyin/mcns_ad_split'

# (pre compartment, post compartment) -> which edge lists carry it. 'b' = not split
BRANCHES = {
    ('split', 'split'): ['aa', 'ad', 'da', 'dd'],
    ('split', 'nosplit'): ['ab', 'db'],
    ('nosplit', 'split'): ['ba', 'bd'],
    ('nosplit', 'nosplit'): ['bb'],
}

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


meta = pd.read_csv('data/maleCNS/mcns_all_neuron_meta.csv', index_col=0, low_memory=False)
meta = meta.rename(columns={"superclass": "super_class", 'class': 'cell_class', 'subclass': 'cell_sub_class'})
in_meta = set(meta.bodyid)
selected = set(meta.bodyid[
    (~meta.super_class.str.contains('sensory|motor'))
    & (meta.cell_class != 'Kenyon_Cell')
    & (~meta.cell_type.isin(['APL', 'DPM']))
])

# ---- who ended up split, and who was left whole ----
split_ids = set()
for sub in ["axon_in", "axon_out", "dendrite_in", "dendrite_out"]:
    split_ids |= set(int(f[:-4]) for f in os.listdir(os.path.join(folder, sub)))
nosplit_ids = set()
for f, col in [('from_nosplit_syn_count.csv', 'body_pre'), ('to_nosplit_syn_count.csv', 'body_post')]:
    nosplit_ids |= set(pd.read_csv(os.path.join(folder, 'syn_count', f), usecols=[col])[col].astype('int64').unique())

# ---- one pass over the synapse table: total and per-branch ground truth ----
truth = collections.Counter()
one_polarity_seen = [set(), set()]  # ids seen as pre, as post
with pa.memory_map(os.path.join(folder, "syn-partners-male-cns-v1.0-minconf-0.5.feather")) as src:
    reader = ipc.open_file(src)
    for i in range(reader.num_record_batches):
        syn = reader.get_batch(i).to_pandas()[['body_pre', 'body_post']]
        syn = syn[(syn.body_pre != syn.body_post) & syn.body_pre.isin(in_meta) & syn.body_post.isin(in_meta)]
        one_polarity_seen[0] |= set(syn.body_pre.unique())
        one_polarity_seen[1] |= set(syn.body_post.unique())
        pre = np.where(syn.body_pre.isin(split_ids), 'split',
                       np.where(syn.body_pre.isin(nosplit_ids), 'nosplit', 'unclassified'))
        post = np.where(syn.body_post.isin(split_ids), 'split',
                        np.where(syn.body_post.isin(nosplit_ids), 'nosplit', 'unclassified'))
        truth.update(zip(pre, post))
total_truth = sum(truth.values())

# ---- what the edge lists actually contain ----
observed = {}
all_pre, all_post = set(), set()
for key, names in BRANCHES.items():
    observed[key] = 0
    for n in names:
        el = pd.read_csv(os.path.join(folder, "result", f"{n}_el.csv"))
        observed[key] += el.syn_count.sum()
        all_pre |= set(el.pre_root_id.astype('int64').unique())
        all_post |= set(el.post_root_id.astype('int64').unique())
        if (el.syn_count <= 0).any():
            failures.append(f"{n}_el.csv has non-positive syn_count")
        if (el.pre_root_id == el.post_root_id).any():
            failures.append(f"{n}_el.csv contains self-edges")
total_observed = sum(observed.values())

print(f"\n{'':<24}{'edge lists':>14}{'synapse table':>16}{'diff':>10}")
for key, names in BRANCHES.items():
    o, t = observed[key], truth[key]
    print(f"{key[0] + ' -> ' + key[1]:<24}{o:>14}{t:>16}{o - t:>+10}")
print(f"{'TOTAL':<24}{total_observed:>14}{total_truth:>16}{total_observed - total_truth:>+10}\n")

check("every synapse counted exactly once", total_observed == total_truth,
      f"{total_observed - total_truth:+d} of {total_truth}")
for key in BRANCHES:
    check(f"{key[0]} -> {key[1]} conserved", observed[key] == truth[key],
          f"{observed[key] - truth[key]:+d}")
check("every synapse's partners are classified", truth[('unclassified', 'split')] == 0
      and truth[('split', 'unclassified')] == 0 and truth[('unclassified', 'unclassified')] == 0
      and truth[('unclassified', 'nosplit')] == 0 and truth[('nosplit', 'unclassified')] == 0)

# ---- the split stage ----
fail_dir = os.path.join(folder, "split_failures")
failed = []
if os.path.isdir(fail_dir):
    for f in os.listdir(fail_dir):
        failed.append(pd.read_csv(os.path.join(fail_dir, f)))
failed = pd.concat(failed) if failed else pd.DataFrame(columns=['bodyid', 'error'])
check("no neuron errored during the split", len(failed) == 0, f"{len(failed)} failed")
if len(failed):
    print(failed.error.value_counts().head(5).to_string())

n_seg = len(os.listdir(os.path.join(folder, "seg_indices")))
one_polarity = one_polarity_seen[0].symmetric_difference(one_polarity_seen[1]) | (
    selected - one_polarity_seen[0] - one_polarity_seen[1])
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
KEY = ['body_pre', 'body_post', 'x', 'y', 'z', 'partner_x', 'partner_y', 'partner_z']
at_risk = set(meta.bodyid[
    meta.super_class.str.contains('motor|efferent|sensory', na=False) | (meta.cell_type == 'IPC')
]) & split_ids
dup = 0
for i in at_risk:
    for a, b in [("axon_in", "dendrite_in"), ("axon_out", "dendrite_out")]:
        pa_, pb = os.path.join(folder, a, f"{i}.csv"), os.path.join(folder, b, f"{i}.csv")
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
