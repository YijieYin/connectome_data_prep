import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import connectome_interpreter as coin 
import seaborn as sns
from tqdm import tqdm

# load activations - some more activated than others ----
act_loss = pd.read_csv('/cephfs2/yyin/tangential/optimised_input/olf_only_activations.csv', index_col=0)
plt.figure()
act_loss.activation.hist()
plt.xlabel('Activation loss (mean activation summed across timesteps)')
plt.ylabel('Number of cell types')
plt.savefig('plots/activation_hist.pdf')

# cell types that couldn't be activated 
act_loss[act_loss.activation == 0]

selected_types = act_loss[act_loss.activation <-0.5]
selected_types.shape

# load data ----
meta = pd.read_csv(
    'data/adult_cb_neuron_meta_no_CX_axonic_postsynapses.csv', index_col=0)
# get glomeruli for ORNs
meta.cell_type[meta.cell_class == 'olfactory'] = meta[meta.cell_class == 'olfactory'].cell_type.str.replace('ORN_','')

idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_root = dict(zip(meta.idx, meta.root_id))
root_to_type = dict(zip(meta.root_id, meta.cell_type))
# visual_projection neurons are considered visual input here, since all optic lobe neurons are removed
idx_to_modality = dict(zip(meta.idx, meta.cell_class))
idx_to_modality.update(dict.fromkeys(meta.idx[meta.super_class == 'visual_projection'], 'visual_projection'))
idx_to_modality.update(dict.fromkeys(meta.idx[meta.super_class == 'ascending'], 'ascending'))

sensory_indices = sorted(list(
    set(meta.idx[meta.cell_class == 'olfactory'])))

# have a look at one type ----
atype = 'CB.FB5E0'
opt_in = np.load('/cephfs2/yyin/tangential/optimised_input/opt_in_olf_only_' + atype + '.npy')
out = np.load('/cephfs2/yyin/tangential/output/out_olf_only_' + atype + '.npy')
opt_in.shape # len(sensory_indices) * 9 

# plot input neuron activation across timesteps ---- 
plt.figure(figsize=(10, 6))
# Use 'imshow' for a 2D heatmap-like visualization
plt.imshow(opt_in, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Input neuron activation')
plt.xlabel('Timesteps')
# plt.ylabel(f'Elements in Column {column_index}')
# plt.title(f'Changes in Column {column_index} During Training')
plt.savefig('plots/all_input_across_time.png')

# plot output neuron activation across timesteps ---- 
plt.figure(figsize=(10, 6))
# Use 'imshow' for a 2D heatmap-like visualization
plt.imshow(out, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Output neuron activation')
plt.xlabel('Timesteps')
# plt.ylabel(f'Elements in Column {column_index}')
# plt.title(f'Changes in Column {column_index} During Training')
plt.savefig('plots/all_activation_across_time.png')

# activation histogram ----
df = opt_in.copy()
plt.figure(figsize=(10, 6))
for i in range(df.shape[1]):
    filtered = np.where(df[:, i] > 0.2)
    if len(filtered[0]) > 0:
        plt.hist(df[filtered[0], i], label=i, alpha=0.4)
    else:
        continue

plt.legend()
plt.savefig('plots/neuron_activation_hist.png')


# get neuron activations ---- 
in_act = coin.utils.get_activations(
    opt_in, sensory_indices, idx_map=idx_to_root)
out_act = coin.utils.get_activations(
    out, list(range(meta.idx.max()+1)), idx_to_root)

# check target neuron activation ---- 
neurons_of_interest = meta.root_id[meta.cell_type == atype]
dfs = []
for i in range(len(out_act)):
    act_layer = pd.DataFrame({n: out_act[i][n] for n in neurons_of_interest}.items(
    ), columns=['neuron', 'activation'])
    act_layer.loc[:, ['timestep']] = i+1
    dfs.append(act_layer)

neuron_activation = pd.concat(dfs, axis = 0)
neuron_activation = neuron_activation.pivot(index = 'neuron', columns = 'timestep', values = 'activation')

# Plotting the heatmap 
plt.figure(figsize=(10, 25))  # You can adjust the size to fit your dataset
# sns.heatmap(neuron_activation.set_index('cell_type'), 
sns.heatmap(neuron_activation, 
            # annot=True, fmt=".2f", 
            cmap="coolwarm", cbar=True)
plt.title('Neuron Activity Heatmap')
plt.xlabel('Timestep')
plt.ylabel('Neuron ID')
plt.savefig('plots/neurons_of_interest.png')

# so what is the best input pattern? ---- 
# get neurons with activation>threshold at any point 
in_act_filtered = coin.utils.get_activations(
    opt_in, sensory_indices, idx_map=idx_to_root, threshold=0.6)
activated = []
for i in range(len(in_act_filtered)): 
    activated.extend(list(in_act_filtered[i].keys()))
activated = set(activated)
len(activated)
# get a df 
dfs = []
for i in range(len(in_act)):
    act_layer = pd.DataFrame({n: in_act[i][n] for n in activated}.items(
    ), columns=['neuron', 'activation'])
    act_layer.loc[:, ['timestep']] = i+1
    dfs.append(act_layer)

neuron_activation = pd.concat(dfs, axis = 0)
neuron_activation = neuron_activation.pivot(index = 'neuron', columns = 'timestep', values = 'activation')

# group by type 
neuron_activation.index = neuron_activation.index.map(root_to_type)
neuron_activation = neuron_activation.groupby('neuron').mean()
neuron_activation.sort_index(inplace=True)

# Plotting the heatmap 
plt.figure(figsize=(10, 25))  # You can adjust the size to fit your dataset
# sns.heatmap(neuron_activation.set_index('cell_type'), 
sns.heatmap(neuron_activation, 
            # annot=True, fmt=".2f", 
            cmap="coolwarm", cbar=True)
plt.title('Neuron Activity Heatmap')
plt.xlabel('Timestep')
plt.ylabel('Neuron ID')
plt.savefig('plots/activated_input_across_time.png')


# take out olfaction and map to chemicals ----
orns = meta.root_id[meta.cell_class == 'olfactory']
dfs = []
for i in range(len(in_act)):
    act_layer = pd.DataFrame({n: in_act[i][n] for n in orns}.items(
    ), columns=['neuron', 'activation'])
    act_layer.loc[:, ['timestep']] = i+1
    dfs.append(act_layer)

neuron_activation = pd.concat(dfs, axis = 0)
neuron_activation = neuron_activation.pivot(index = 'neuron', columns = 'timestep', values = 'activation')

# group by type 
neuron_activation.index = neuron_activation.index.map(root_to_type)
neuron_activation = neuron_activation.groupby('neuron').mean()
neuron_activation.sort_index(inplace=True)

# Plotting the heatmap 
plt.figure(figsize=(10, 25))  # You can adjust the size to fit your dataset
# sns.heatmap(neuron_activation.set_index('cell_type'), 
sns.heatmap(neuron_activation, 
            # annot=True, fmt=".2f", 
            cmap="coolwarm", cbar=True)
plt.title('Neuron Activity Heatmap')
plt.xlabel('Timestep')
plt.ylabel('Neuron ID')
plt.savefig('plots/orn_activation_across_time.png')

# cosine similarity ----
sim = coin.external_map.map_to_experiment(neuron_activation, 'Dweck_adult_fruit')
sim.idxmax(axis = 1)

# choose the cell type and chemicals to inspect 'raw' data 
ctype = [1,2,3]
chem = ['Strawberry','African breadfruit','Java plum']
ex_data = coin.external_map.load_dataset('Dweck_adult_fruit')
neuron_activation.index.name = 'glomerulus'
df_dsp = neuron_activation.loc[:, list(set(ctype))].join(ex_data[list(set(chem))], how='inner', on='glomerulus')


# focus on the visual input ---- 
vpns = meta.root_id[meta.super_class == 'visual_projection']
in_act_filtered = coin.utils.get_activations(
    opt_in, sensory_indices, idx_map=idx_to_root, threshold=0.1)
activated = []
for i in range(len(in_act_filtered)): 
    activated.extend(list(in_act_filtered[i].keys()))
activated = set(activated)
activated = activated & set(vpns)

dfs = []
for i in range(len(in_act)):
    act_layer = pd.DataFrame({n: in_act[i][n] for n in activated}.items(
    ), columns=['neuron', 'activation'])
    act_layer.loc[:, ['timestep']] = i+1
    dfs.append(act_layer)

neuron_activation = pd.concat(dfs, axis = 0)
neuron_activation = neuron_activation.pivot(index = 'neuron', columns = 'timestep', values = 'activation')

neuron_activation.index = neuron_activation.index.map(root_to_type)
neuron_activation = neuron_activation.groupby('neuron').mean()
neuron_activation.sort_index(inplace=True)

# Plotting the heatmap 
plt.figure(figsize=(10, 25))  # You can adjust the size to fit your dataset
# sns.heatmap(neuron_activation.set_index('cell_type'), 
sns.heatmap(neuron_activation, 
            # annot=True, fmt=".2f", 
            cmap="coolwarm", cbar=True)
plt.title('Neuron Activity Heatmap')
plt.xlabel('Timestep')
plt.ylabel('Neuron ID')
plt.savefig('plots/vpn_activation_across_time.png')

# make flywire url 
import requests
import re
def shorten_url(long_url):
   shortener = "https://shortng-bmcp5imp6q-uc.a.run.app/shortng"
   r = requests.post(shortener, data={"text": long_url, "client": "web", "filename": None, "title": None})
   r.raise_for_status()
   return re.search("<a href=(.*?json)>", r.content.decode()).group(1)

url = coin.utils.get_ngl_link(neuron_activation[neuron_activation.index.isin(meta.root_id[meta.cell_type == 'LC13'])])
shorten_url(url)

# dimensionality reduction ----
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
import matplotlib
import matplotlib.cm as cm

orns = meta.cell_type[meta.cell_class == 'olfactory'].unique()
vecs = []
for atype in tqdm(selected_types.cell_type): 
    opt_in = np.load('/cephfs2/yyin/tangential/optimised_input/opt_in_olf_only_' + atype + '.npy')
    # get neuron activations in glomeruli 
    in_act = coin.utils.get_activations(
        opt_in, sensory_indices, idx_map=idx_to_type)
    # turn into a vector 
    vec = []
    for i in range(len(in_act)): 
        vec.extend([in_act[i][n] for n in orns])
    vecs.append(vec)

distances = pdist(np.array(vecs), metric='euclidean')
# Convert to a square matrix form
dist_matrix = squareform(distances)

# Perform hierarchical clustering
Z = linkage(distances, method='ward')
cl = fcluster(Z, 3, criterion ='maxclust')
type2cl = {id:c for id,c in zip(selected_types.cell_type, cl)}
selected_types.loc[:,['cluster']] = selected_types.cell_type.map(type2cl)
selected_types.loc[:,['norm_activation']] = -selected_types.activation.copy()
selected_types.loc[:,['norm_activation']] = (selected_types.norm_activation - selected_types.norm_activation.min()) / (selected_types.norm_activation.max() - selected_types.norm_activation.min())
label2col = dict(zip(selected_types.cell_type, selected_types.norm_activation))

# Plot the dendrogram
palette = sns.color_palette().as_hex()
set_link_color_palette(list(palette))
# set linewidth 
matplotlib.rcParams['lines.linewidth'] = 3

plt.figure(figsize=(10, 3))
dendrogram(Z, labels = selected_types.cell_type.unique(), color_threshold = 2.5)

ax = plt.gca()
x_labels = ax.get_xmajorticklabels()

# Change the label colors based on the label_to_color dictionary
for lbl in x_labels:
    lbl.set_color(cm.viridis(label2col[lbl.get_text()]))
plt.savefig('plots/olf_dend.pdf')