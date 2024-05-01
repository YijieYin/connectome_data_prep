import numpy as np
import pandas as pd
import scipy as sp
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import connectome_interpreter as coin
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


inprop = sp.sparse.load_npz(
    'data/adult_inprop_cb_neuron_no_tangential_postsynapses_in_CX.npz')
meta = pd.read_csv(
    'data/adult_cb_neuron_meta_no_tangential_postsynapses_in_CX.csv', index_col=0)
# inprop = sp.sparse.load_npz('data/adult_type_inprop.npz')
# meta = pd.read_csv('data/adult_type_meta.csv', index_col=0)

idx_to_type = dict(zip(meta.idx, meta.cell_type))
idx_to_cell_class = dict(zip(meta.idx, meta.cell_class))
idx_to_sign = dict(zip(meta.idx, meta.sign))
idx_to_root = dict(zip(meta.idx, meta.root_id))
root_to_type = dict(zip(meta.root_id, meta.cell_type))

num_layers = 5
target_indices = meta.idx[meta.cell_class == 'Kenyon_Cell'].unique()

if meta.idx.isna().any():
    print('Some cell types are not taken into account: there are NAs in the indices in meta.')
    meta = meta[~meta.idx.isna()]
meta.idx = meta.idx.astype(int)

# update the connectivity as necessary
# remove self connections by cells within a type
# df = pd.DataFrame({'input_idx': range(meta.idx.max()),
#                 'output_idx': range(meta.idx.max()),
#                 'value': 0})
# inprop = coin.utils.modify_coo_matrix(inprop, updates_df=df)
# remove connections from KCs and DANs to KCs, since they are axo-axonic
inprop = coin.utils.modify_coo_matrix(inprop, input_idx=meta.idx[meta.cell_class.isin(['Kenyon_Cell', 'DAN'])].unique(),
                                      output_idx=meta.idx[meta.cell_class == 'Kenyon_Cell'].unique(), value=0)
inprop = coin.utils.modify_coo_matrix(inprop,
                                      set(meta.idx[meta.cell_class ==
                                          'Kenyon_Cell']),
                                      set(meta.idx[meta.cell_class == 'DAN']),
                                      0)
# sp.sparse.save_npz('C:\\Users\\44745\\projects\\interpret_connectome\\data\\adult_inprop.npz', inprop)

# add negative connections
idx_to_sign = dict(zip(meta.idx, meta.sign))
inprop_dense = np.array(inprop.todense(), dtype=np.float32)

neg_indices = [idx for idx, val in idx_to_sign.items() if val == -1]
inprop_dense[neg_indices] = -inprop_dense[neg_indices]

# inprop_tensor = torch.from_numpy(inprop_dense).t().to(device)

# make model
# sorting sensory_indices is important because it makes the ordering of indices reproducible down the line
# set() doesn't preserve the order
sensory_indices = sorted(list(
    set(meta.idx[meta.super_class.isin(['sensory', 'visual_projection', 'ascending'])])))
# sensory_indices = sorted(list(set(meta.idx[meta.cell_class == 'ALPN'])))
non_sensory_indices = [i for i in range(
    meta.idx.max()+1) if i not in sensory_indices]
ml_model = coin.activation_maximisation.MultilayeredNetwork(
    torch.from_numpy(inprop_dense).float().t().to(device),
    sensory_indices, threshold=0, tanh_steepness=5, num_layers=num_layers).to(device)


def regularisation(tensor):
    return torch.norm(tensor, 1)


# target_indices = list(map(int, target_indices.split(',')))
# activate target neurons in every layer
# target_index_dic = {i: target_indices for i in range(num_layers)}
# activate target neurons in the last layer
target_index_dic = {num_layers-1: target_indices}
bump1 = meta.idx[(meta.side == 'left') & (
    meta.ito_lee_hemilineage == 'DM2_CX_d2') & (meta.cell_type == 'EPG')].tolist()
bump2 = meta.idx[(meta.side == 'right') & (
    meta.ito_lee_hemilineage == 'DM3_CX_d2') & (meta.cell_type == 'EPG')].tolist()
bump3 = meta.idx[(meta.side == 'left') & (
    meta.ito_lee_hemilineage == 'DM2_CX_d1') & (meta.cell_type == 'EPG')].tolist()
target_index_dic = dict.fromkeys(range(6), bump1)
target_index_dic.update(dict.fromkeys(range(6, 9, 1), bump2))
target_index_dic.update(dict.fromkeys(range(9, 12, 1), bump3))

opt_in, out, act_loss, out_reg_loss, in_reg_los, snapshots = coin.activation_maximisation.activation_maximisation(ml_model,
                                                                                                                  target_index_dic,
                                                                                                                  num_iterations=50, learning_rate=0.4,
                                                                                                                  in_regularisation_lambda=3e-4, custom_in_regularisation=regularisation,
                                                                                                                  out_regularisation_lambda=0.2,
                                                                                                                  device=device,
                                                                                                                  stopping_threshold=1e-6,
                                                                                                                  wandb=False)

# or load from files
opt_in = np.load(
    '/cephfs2/yyin/moving_bump/optimised_input/opt_in_4664990.npy')
out = np.load('/cephfs2/yyin/moving_bump/output/out_4664990.npy')

# plot network
paths = coin.activation_maximisation.activations_to_df(
    inprop_dense, opt_in, out, sensory_indices,
    inidx_mapping=idx_to_root,
    activation_threshold=0.3, connectivity_threshold=0.01)

# optionally remove the ones that don't lead from input to output
paths_f = coin.path_finding.remove_excess_neurons(paths,
                                                  #   keep=meta.cell_class[meta.idx.isin(
                                                  #       sensory_indices)],
                                                  target_indices=meta.root_id[meta.idx.isin(bump1 + bump2 + bump3)])
coin.utils.plot_layered_paths(paths_f, figsize=(10, 20),
                              # choose cells to be put on top
                              priority_indices=meta.cell_type[meta.idx.isin(
                                  sensory_indices)],
                              sort_by_activation=True)


# ----------------------------------------------
# get activations by idx_to_x mapping
in_act = coin.utils.get_activations(
    opt_in, sensory_indices, idx_map=idx_to_type)
out_act = coin.utils.get_activations(
    out, list(range(meta.idx.max()+1)), idx_to_root)

neurons_of_interest = meta.root_id[meta.idx.isin(
    bump1 + bump2 + bump3)].tolist()
neurons_of_interest.extend(meta.root_id[meta.cell_type.isin(
    ['Delta7']) | meta.cell_sub_class.isin(['ring neuron'])].tolist())
neurons_of_interest.extend(meta.root_id[meta.cell_type.isin(
    ['AN_IPS_LAL_1', 'MeTu1', 'MeTu2', 'MeTu3', 'MeTu4']) & (meta.side == 'left')].tolist())
dfs = []
for i in range(len(out_act)):
    act_layer = pd.DataFrame({n: out_act[i][n] for n in neurons_of_interest}.items(
    ), columns=['neuron', 'activation'])
    act_layer.loc[:, ['timestep']] = i+1
    dfs.append(act_layer)

neuron_activation = pd.concat(dfs, axis=0)
neuron_activation = neuron_activation.pivot(
    index='neuron', columns='timestep', values='activation')
url = coin.utils.get_ngl_link(neuron_activation, colour_saturation=0.1)
df = pd.DataFrame.from_dict({'url': [url]})
df.url.to_clipboard(index=False, header=False)
df.to_csv('export_url.csv')

neuron_activation.loc[:, ['cell_type']
                      ] = neuron_activation.index.map(root_to_type)
neuron_activation.sort_values(['cell_type'], inplace=True)

# Plotting the heatmap ----
plt.figure(figsize=(10, 100))  # You can adjust the size to fit your dataset
sns.heatmap(neuron_activation.set_index('cell_type'),
            # annot=True, fmt=".2f",
            cmap="coolwarm", cbar=True)
plt.title('Neuron Activity Heatmap')
plt.xlabel('Timestep')
plt.ylabel('Neuron ID')
plt.savefig('plots/neurons_of_interest.png')

# ------------------------------------------------------------------------------------------------
# plot snapshots across training
layer = 3
snapshot_viz = []
for snapshot in snapshots:
    snapshot_viz.append(snapshot[:, layer])
snapshot_viz = np.stack(snapshot_viz, axis=-1)

plt.figure(figsize=(10, 6))
# Use 'imshow' for a 2D heatmap-like visualization
plt.imshow(snapshot_viz, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Input neuron activation')
plt.xlabel('Snapshots through training')
# plt.ylabel(f'Elements in Column {column_index}')
# plt.title(f'Changes in Column {column_index} During Training')
plt.show()

# -------------------------------------------------
# and histograms
df = out.copy()
plt.figure(figsize=(10, 6))
for i in range(df.shape[1]):
    filtered = np.where(df[:, i] > 0.2)
    if len(filtered[0]) > 0:
        plt.hist(df[filtered[0], i], label=i, alpha=0.4)
    else:
        continue

plt.legend()
plt.savefig('plots/neuron_activation_hist.png')

# ----------------------------------------------
# plot input neuron activation across timesteps
plt.figure(figsize=(10, 6))
# Use 'imshow' for a 2D heatmap-like visualization
plt.imshow(opt_in, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Input neuron activation')
plt.xlabel('Timesteps')
# plt.ylabel(f'Elements in Column {column_index}')
# plt.title(f'Changes in Column {column_index} During Training')
plt.savefig('plots/all_input_across_time.png')

plt.show()


# plot output neuron activation across timesteps
plt.figure(figsize=(10, 6))
# Use 'imshow' for a 2D heatmap-like visualization
plt.imshow(out, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Output neuron activation')
plt.xlabel('Timesteps')
# plt.ylabel(f'Elements in Column {column_index}')
# plt.title(f'Changes in Column {column_index} During Training')
plt.savefig('plots/all_activation_across_time.png')

plt.show()

# ----------------------------------------------
# manipulate the input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inprop_tensor = torch.from_numpy(inprop_dense).t().to(device)
# make model
ml_model = coin.activation_maximisation.MultilayeredNetwork(
    inprop_tensor, sensory_indices, threshold=0, tanh_steepness=5, num_layers=num_layers).to(device)

# manipulate a particular group
global_idx = meta.idx[meta.cell_type == 'TRN_VP2'].values
local_idx = [i for i, j in enumerate(sensory_indices) if j in global_idx]
opt_inm = opt_in.copy()
opt_inm[local_idx, 0] = 1

# or manipuate the whole thing
opt_inm = np.zeros_like(opt_in)
opt_inm[:, 0] = opt_in[:, 0]
# end of manipulation

input_tensor = torch.from_numpy(opt_inm).to(device)
outm = ml_model(input_tensor).cpu().detach().numpy()
out_act = coin.utils.get_activations(
    outm, non_sensory_indices, idx_to_type)

out_act_before = []
for i in range(len(out_act)):
    df = pd.DataFrame(out_act[i].items(), columns=['cell_type', 'activation'])
    df['step'] = i
    out_act_before.append(df)
out_act_before = pd.concat(out_act_before)
out_act_before[out_act_before.cell_type.str.contains('KC')].activation.sum()

out_act_after = []
for i in range(len(out_act)):
    df = pd.DataFrame(out_act[i].items(), columns=['cell_type', 'activation'])
    df['step'] = i
    out_act_after.append(df)
out_act_after = pd.concat(out_act_after)
out_act_after[out_act_after.cell_type.str.contains('KC')].activation.sum()

# ----------------------------------------------
# for a one-layered network, look at how inhibition/excitation interact with the results of activation maximisation
# PN -> LHLNs
top_in = coin.utils.get_activations(
    opt_in, sensory_indices, idx_to_type)
pnlhln = coin.compress_paths.result_summary(inprop_dense,
                                            inidx=sensory_indices,
                                            outidx=target_indices,
                                            inidx_map=idx_to_type, display_output=False)
pnlhln_edgelist = pnlhln.stack().reset_index()
pnlhln_edgelist.columns = ['Source', 'Target', 'Weight']
pnlhln_edgelist = pnlhln_edgelist[pnlhln_edgelist['Weight'] != 0]
pnlhln_edgelist['sign'] = ['excitatory' if w >
                           0 else 'inhibitory' for w in pnlhln_edgelist['Weight']]
# have columns 'excitatory' and 'inhibitory'
pnlhln_edgelist = pnlhln_edgelist.pivot_table(
    index=['Source', 'Target'], columns='sign', values='Weight', fill_value=0).reset_index()
pnlhln_edgelist['activation'] = pnlhln_edgelist['Source'].map(top_in[0])
fig = px.scatter(pnlhln_edgelist, x='excitatory', y='inhibitory',
                 color='activation',  # This column should be normalized between 0 and 1
                 color_continuous_scale='Viridis',  # Using the Viridis color map
                 hover_name=pnlhln_edgelist.index)  # Display index on hover

fig.show()

# ----------------------------------------------
# compare effective connectivity (not signed) with activation of input neurons
step = sp.sparse.load_npz(
    'C:\\Users\\44745\\Downloads\\fafb_type_step_1.npz')
us = coin.compress_paths.result_summary(step,
                                        # outidx=13702,
                                        outidx=target_indices,
                                        inidx=sensory_indices,
                                        inidx_map=idx_to_type, outidx_map=idx_to_cell_class, display_output=False)
us['activation'] = us.index.map(top_in[0])

us.sort_values('Kenyon_Cell', ascending=False)
# add sign of connectivity
type_to_sign = dict(zip(meta.cell_type, meta.sign))
us['sign'] = us.index.map(type_to_sign)
us.sort_values('Kenyon_Cell', ascending=False)

# ---------------------------
# for a two-layered network, explore how excitation/inhibition interact with the results of activation maximisation
inkc_1 = pd.read_csv(
    'C:\\Users\\44745\\Downloads\\input_to_kc_1.csv', index_col=0)
inkc_1 = inkc_1.groupby('pre')[['excitatory', 'inhibitory']].mean()
us = us.merge(inkc_1, left_index=True, right_index=True)
us['pre'] = us.index
# normalise the 'activation' column
us['activation_norm'] = (us.activation - us.activation.min()) / \
    (us.activation.max() - us.activation.min())

plt.scatter(us.excitatory, us.inhibitory, alpha=us.activation)
plt.show()

fig = px.scatter(us, x='excitatory', y='inhibitory',
                 color='activation_norm', hover_data='pre')
fig.show()

# ----------------------------------------------
# how activation of input neurons changes during optimisation; compared to the effective connectivity
for i, snap in enumerate(snapshots):
    group_activation_map = coin.utils.get_activations(
        snap, sensory_indices, idx_to_type)
    us['activation_step_'+str(i)] = us.index.map(group_activation_map[0])

# change of activation across training steps
us_melted = us.melt(id_vars=['Kenyon_Cell', 'sign'], value_vars=[f'activation_step_{i}' for i in range(10)],
                    var_name='activation_step', value_name='activation_value')

# Plot
plt.figure(figsize=(10, 6))
colors = {1: 'blue', -1: 'red'}

for step in range(10):
    step_df = us_melted[us_melted['activation_step']
                        == f'activation_step_{step}']
    plt.scatter(step_df['Kenyon_Cell'], step_df['activation_value'],
                color=step_df['sign'].map(colors), alpha=(step+1)/10, label=f'Snapshot {step}')

plt.xlabel('connectivity to target neruons from input neurons')
plt.ylabel('activation of input neurons')
plt.legend()
plt.show()
