from connectome_interpreter import *
import time
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

times = {}
dense_times = {}
dense_cpu_times = {}
sizes = {}

device = 'cpu'

# ---- larva ---- 
inprop = sp.sparse.load_npz("../data/larva/larva_inprop.npz")
time_start = time.time()
steps = compress_paths(inprop, 5)
time_end = time.time()
times['larva'] = time_end - time_start
sizes['larva'] = inprop.shape[0]
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop, 5)
time_end = time.time()
dense_times['larva'] = time_end - time_start
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop, 5, device=device)
time_end = time.time()
dense_cpu_times['larva'] = time_end - time_start
del inprop, steps
print(f'larva done, size {sizes["larva"]}, time {times["larva"]}, dense {dense_times["larva"]}, dense cpu {dense_cpu_times["larva"]}')

# ---- hemibrain ----
inprop = sp.sparse.load_npz("../data/hemibrain/hemibrain_neuron.npz")
time_start = time.time()
steps = compress_paths(inprop, 5)
time_end = time.time()
times["hemibrain"] = time_end - time_start
sizes["hemibrain"] = inprop.shape[0]
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop.astype("float32"), 5)
time_end = time.time()
dense_times['hemibrain'] = time_end - time_start
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop.astype("float32"), 5, device=device)
time_end = time.time()
dense_cpu_times['hemibrain'] = time_end - time_start
del inprop, steps
print(f'hemibrain done, size {sizes["hemibrain"]}, time {times["hemibrain"]}, dense {dense_times["hemibrain"]}, dense cpu {dense_cpu_times["hemibrain"]}')

# ---- manc ----
inprop = sp.sparse.load_npz("../data/manc_inprop.npz")
time_start = time.time()
steps = compress_paths(inprop, 5)
time_end = time.time()
times["manc"] = time_end - time_start
sizes["manc"] = inprop.shape[0]
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop, 5)
time_end = time.time()
dense_times['manc'] = time_end - time_start
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop, 5, device=device)
time_end = time.time()
dense_cpu_times['manc'] = time_end - time_start
del inprop, steps
print(f'manc done, size {sizes["manc"]}, time {times["manc"]}, dense {dense_times["manc"]}, dense cpu {dense_cpu_times["manc"]}')

# ---- fafb central brain ---- 
inprop = sp.sparse.load_npz("../data/adult_cb_neuron/adult_inprop_cb_neuron.npz")
time_start = time.time()
steps = compress_paths(inprop, 5)
time_end = time.time()
times["fafb_cb"] = time_end - time_start
sizes["fafb_cb"] = inprop.shape[0]
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop, 5)
time_end = time.time()
dense_times['fafb_cb'] = time_end - time_start
del inprop, steps
# make dense_cpu_time dummy: na
dense_cpu_times['fafb_cb'] = pd.NA
print(f'fafb_cb done, size {sizes["fafb_cb"]}, time {times["fafb_cb"]}, dense {dense_times["fafb_cb"]}')

# ---- fafb optic lobe ---- 
inprop = sp.sparse.load_npz("../data/fafb_optic_right_neuron/fafb_inprop_optic_right_neuron.npz")
time_start = time.time()
steps = compress_paths(inprop, 5)
time_end = time.time()
times["fafb_optic"] = time_end - time_start
sizes["fafb_optic"] = inprop.shape[0]
del steps

time_start = time.time()
steps = compress_paths_not_chunked(inprop, 5)
time_end = time.time()
dense_times['fafb_optic'] = time_end - time_start
del inprop, steps
dense_cpu_times['fafb_optic'] = pd.NA
print(f'fafb_optic done, size {sizes["fafb_optic"]}, time {times["fafb_optic"]}, dense {dense_times["fafb_optic"]}')

# ---- fafb all neuron ----
# skip dense matmul because 139116*139116*32/8/1e9 = 77GB, too big 
inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_inprop_all_neuron.npz")
time_start = time.time()
steps = compress_paths(inprop, 5)
time_end = time.time()
times["fafb_all"] = time_end - time_start
sizes["fafb_all"] = inprop.shape[0]
del inprop, steps
dense_times['fafb_all'] = pd.NA
dense_cpu_times['fafb_all'] = pd.NA

# make dataframe 
connectome_result = pd.DataFrame({"my_time": times, "matrix_size": sizes, "gpu_time": dense_times, "cpu_time": dense_cpu_times})
connectome_result = connectome_result.melt(
    id_vars = 'matrix_size',
    value_vars = ['my_time','gpu_time', 'cpu_time'], ignore_index=False,
    var_name = 'method'
).rename(columns={'value': 'time'})


# ---- random matrix matching density ---- 
gpu_times = []
cpu_times = []
my_times = []
sizes = [100, 1000, 3000, 5000, 10000, 15_000, 20_000]

inprop = sp.sparse.load_npz("../data/fafb_all_neuron/fafb_inprop_all_neuron.npz")
# calculate density
density = inprop.nnz / (inprop.shape[0] * inprop.shape[1])

for matsize in sizes: 
    # make a random matrix with the density above, element float32
    rand_mat = sp.sparse.random(matsize, matsize, density=density, format="csr", dtype=np.float32)
    time_start = time.time()
    steps = compress_paths_not_chunked(
        rand_mat, 5, device='cuda'
    )
    time_end = time.time()
    del steps
    gpu_times.append(time_end - time_start)

    time_start = time.time()
    steps = compress_paths_not_chunked(rand_mat, 5, device="cpu")
    time_end = time.time()
    cpu_times.append(time_end - time_start)
    del steps

    time_start = time.time()
    steps = compress_paths(rand_mat, 5)
    time_end = time.time()
    my_times.append(time_end - time_start)
    del steps, rand_mat

randmat_result = pd.DataFrame({
    "matrix_size": sizes,
    'gpu_time': gpu_times,
    'cpu_time': cpu_times,
    'my_time': my_times
}).melt(
    id_vars=["matrix_size"],
    value_vars=["gpu_time", "cpu_time", "my_time"],
    var_name="method", 
    value_name="time"
)

out = (
    pd.concat([connectome_result, randmat_result])
    .reset_index()
    .rename(columns={"index": "connectome"})
)
# replace number-only entries with na from 'connectome'
out.loc[out["connectome"].astype(str).str.isnumeric(), "connectome"] = pd.NA

out = out.replace(
    {
        "gpu_time": "default: GPU",
        "cpu_time": "default: CPU",
    }
)
out.to_csv("matmul_benchmark.csv")


# fit a * x^3:
coefs = {}
for method, sub in out.groupby("method"):
    x3 = sub["matrix_size"] ** 3
    y = sub["time"]
    a, b = np.polyfit(x3, y, 1)  # slope a, intercept b
    coefs[method] = (a, b)


sns.scatterplot(
    data=out,
    x="matrix_size",
    y="time",
    hue="method",
)
# add predicted lines in dash
for method in out.method.unique():
    a, b = coefs[method]
    x = np.linspace(out["matrix_size"].min(), out["matrix_size"].max(), 100)
    y = a * (x ** 3) + b
    plt.plot(x, y, linestyle="--", label=f"Fitted: {method}")
plt.legend()

# y label:
plt.ylabel("Time (seconds)")
# y range
plt.ylim(0, 150)
# plt.yscale('log')
# plt.xscale('log')
plt.savefig("matmul_benchmark.png")
plt.savefig('matmul_benchmark.pdf')