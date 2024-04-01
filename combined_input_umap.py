import numpy as np
import pandas as pd
import scipy as sp
import umap
import matplotlib.pyplot as plt
import torch
import plotly.express as px
from sklearn.cluster import KMeans


import connectome_interpreter as coin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

meta = pd.read_csv('data/adult_type_meta.csv', index_col=0)
inprop = sp.sparse.load_npz('data/adult_type_inprop.npz')
idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_cell_class = dict(zip(meta.idx, meta.cell_class))
idx_to_sign = dict(zip(meta.idx, meta.sign))
inprop_dense = inprop.toarray()
neg_indices = [idx for idx, val in idx_to_sign.items() if val == -1]
inprop_dense[neg_indices] = -inprop_dense[neg_indices]

# all_in = np.load('C:\\Users\\44745\\Downloads\\optimised_inputs_together.npy')
all_in = np.load('optimised_inputs.npy')
all_out = np.load('outputs.npy')
all_in.shape

sensory = True
layer_num = 5

# sorting sensory_indices is important because it makes the ordering of indices reproducible down the line
sensory_indices = sorted(list(
    set(meta.idx[meta.super_class.isin(['sensory', 'visual_projection', 'ascending'])])))
non_sensory_indices = [i for i in range(
    meta.idx.max()+1) if i not in sensory_indices]

#### first take a look at one ####
one_in = all_in[0].reshape(-1, layer_num)
one_out = all_out[0].reshape(-1, layer_num)

# histogram of neuron activation values for each layer
array = one_in
plt.hist([array[:, layer] for layer in range(array.shape[1])], label=[
         f'time point {layer}' for layer in range(array.shape[1])], alpha=0.3)
# for layer in range(array.shape[1]):
#     plt.hist(array[:, layer], label=f'time point {layer}', alpha=0.3)
plt.xlabel('Neuron activation value')
plt.ylabel('Number of neurons')
plt.title('Histogram of non-input neuron activation values for time point')
plt.legend()
plt.show()

top_in = coin.utils.top_n_activated(
    one_in, sensory_indices, idx_to_type, threshold=0.5)
top_out = coin.utils.top_n_activated(
    one_out, non_sensory_indices, idx_to_type, threshold=0.9)

# check against direct connectivity (inprop_dnese)
us = coin.compress_paths.result_summary(inprop_dense,
                                        outidx=13702,
                                        inidx=sensory_indices,
                                        inidx_map=idx_to_type, outidx_map=idx_to_cell_class, display_output=False)
us.sort_values(us.columns[0], ascending=False)
us['activation'] = us.index.map(top_in[7])
us.sort_values('activation', ascending=False)


plt.figure(figsize=(10, 6))
# Use 'imshow' for a 2D heatmap-like visualization
plt.imshow(one_out, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Value')
plt.xlabel('Snapshot Index')
# plt.ylabel(f'Elements in Column {column_index}')
# plt.title(f'Changes in Column {column_index} During Training')
plt.show()


#### Now look at all the inputs with umap ####
data = all_in if sensory else all_out
reducer = umap.UMAP()
embedding = reducer.fit_transform(data)
embedding.shape

# Convert UMAP embeddings to a DataFrame
df_embeddings = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])

# Add a column for row indices to identify rows on hover
df_embeddings['Row'] = np.arange(len(df_embeddings))

plt.scatter(
    embedding[:, 0],
    embedding[:, 1])
plt.show()

# assign clusters
# with kmeans
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(embedding)
df_embeddings['Cluster'] = clusters

fig = px.scatter(df_embeddings, x='UMAP1', y='UMAP2',
                 color='Cluster', hover_data=['Cluster', 'Row'])
fig.show()

# calculate the mean per cluster
df = pd.DataFrame(data)
df['cluster'] = clusters

# Step 2: Calculate and Compare Means for Each Cluster
cluster_means = df.groupby('cluster').mean()
cluster_means = cluster_means.T
cluster_means['idx'] = np.array(sensory_indices)[
    :, np.newaxis].repeat(layer_num, axis=1).flatten()
cluster_means['type'] = cluster_means['idx'].map(idx_to_type)
cluster_means['diff01'] = cluster_means[0] - cluster_means[1]
cluster_means['diff02'] = cluster_means[0] - cluster_means[2]
cluster_means['diff12'] = cluster_means[1] - cluster_means[2]
cluster_means.sort_values('diff02', ascending=False)

# how many neurons have different activations?
# filter out the not so active ones roughly
plt.hist(cluster_means.loc[cluster_means[0].abs() > 0.1, 'diff01'], bins=50)
plt.yscale('log')
plt.xlabel('Difference in activation between cluster 0 and 1')
plt.ylabel('Number of neuron types')
plt.show()
