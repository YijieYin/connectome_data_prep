# interpret_connectome
This repo has the code and some of the output for making use of connectomics data. The `data` folder has data from papers that I have used. 

## Code 
- Files named with `.*prepare_connectome.*` has the scripts on processing raw connectomics data into a `scipy.sparse.matrix` and a meta `.csv` file, for each connectomics dataset. 
- For running activation maximisation on high performance computers, use the `activation_maximisation.py`, and `run_act_max_hpc.sh` scripts.
- `tangential_analysis.py` contains analysis for tangential neuron activation in response to olfactory receptor neuron manipulations.

## Data 
All pre-processed data are in the `data` folder. 
- Adult:
  - Full Adult Fly Brain:  combines type information from [Schlegel et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546055v2) and connectivity information from [Dorkenwald et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546656v2) and [Buhmann et al. 2021](https://www.nature.com/articles/s41592-021-01183-7), and neurotransmitter data from [Eckstein et al. 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6). By using the connectivity information, you agree to follow the [FlyWire citation guidelines and principles](https://codex.flywire.ai/api/download):
    - `adult_type_inprop.npz` & `adult_type_meta.csv`: the central-brain-only connectome, grouped by types from [Schlegel et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546055v2). ~20k*20k; 
    - `adult_inprop_cb_neuron.npz` & `adult_cb_neuron_meta.csv`: the central-brain-only connectome, on a single neuron level. ~50k*50k;
    - `adult_inprop_cb_neuron_no_tangential_postsynapses_in_CX.npz` & `adult_cb_neuron_meta_no_CX_axonic_postsynapses.csv`: the central-brain-only connectome, on a single neuron level. Removed many postsynapses in axonic regions, and presynapses in dendritic regions, for the central complex neurons. ~50*50k. 
- Larva: the processed version of the larval connectivity is `larva_inprop.npz`. Both the cell type annotaitons and the connectivity came from [Winding et al. 2023](https://www.science.org/doi/10.1126/science.add9330). 
