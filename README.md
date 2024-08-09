# interpret_connectome
This repo contains the pre-processed publicly-available connectomics data (in `data` folder), and the code that generated them; as well as some early snippets of analysis. For more extensive analysis, see [this repo](https://github.com/YijieYin/connectome_interpreter). 

## Data 
All pre-processed data are in the `data` folder, typically composed of a `scipy.sparse.matrix` and a meta `.csv` file. 
- Adult:
  - Full Adult Fly Brain (FAFB):  combines type information from [Schlegel et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546055v2) and connectivity information from [Dorkenwald et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546656v2) and [Buhmann et al. 2021](https://www.nature.com/articles/s41592-021-01183-7), and neurotransmitter data from [Eckstein et al. 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6). By using the connectivity information, you agree to follow the [FlyWire citation guidelines and principles](https://codex.flywire.ai/api/download). To explore the dataset in detail in a cool interface, you can go here: [https://tinyurl.com/flywire783](https://tinyurl.com/flywire783):
    - central brain: 
      - `adult_inprop_cb_neuron.npz` & `adult_cb_neuron_meta.csv`: the central-brain-only connectome, on a single neuron level. ~50k*50k;
      - `adult_inprop_cb_neuron_no_tangential_postsynapses_in_CX.npz` & `adult_cb_neuron_meta_no_CX_axonic_postsynapses.csv`: the central-brain-only connectome, on a single neuron level. Removed many postsynapses in axonic regions, and presynapses in dendritic regions, for the central complex neurons. ~50*50k.
      - `adult_type_inprop.npz` & `adult_type_meta.csv`: the central-brain-only connectome, grouped by types from [Schlegel et al. 2023](https://www.biorxiv.org/content/10.1101/2023.06.27.546055v2). ~20k*20k;
    - optic lobe:
      - `fafb_inprop_optic_right_neuron.npz` & `fafb_optic_right_neuron_meta.csv`: single neuron, ~50k*50k. 
  - Male Adult Nerve Cord (MANC): data from [Takemura et al. 2024](https://elifesciences.org/reviewed-preprints/97769) and [Marin et al. 2024](https://elifesciences.org/reviewed-preprints/97766v1). Also publicly available on [neuprint](https://neuprint.janelia.org/?dataset=manc:v1.2.1&qt=findneurons). The data was downloaded using the script `manc_get_connectivity.Rmd`.
    - `manc_inprop.npz` & `manc_meta.csv`: single neuron level, ~23k*23k.
  - male central nervous system (maleCNS) optic lobe: from [Nern et al. 2024](https://www.biorxiv.org/content/10.1101/2024.04.16.589741v2), also publicly available on [neuprint](https://neuprint.janelia.org/?dataset=optic-lobe%3Av1.0&qt=findneurons).
    - `neuprint_inprop_optic.npz` & `neuprint_meta_optic.csv`: single neuron level, ~50k*50k.  
- Larva: the processed version of the larval connectivity is `larva_inprop.npz`. Both the cell type annotaitons and the connectivity came from [Winding et al. 2023](https://www.science.org/doi/10.1126/science.add9330). 

## Code 
- Files named with `.*prepare_connectome.*` has the scripts on processing raw connectomics data into a `scipy.sparse.matrix` and a meta `.csv` file, for each connectomics dataset. Code for downloading of the original published datasets can be found in the respective scripts. 
- For calculating "effective connectivity" (more explanation [here](https://connectome-interpreter.readthedocs.io/en/latest/tutorials/matmul.html)) on high performance computers, use the `matmul_hpc.py` and `matmul_hpc.sh` scripts. 
- For running activation maximisation (more explanation [here](https://connectome-interpreter.readthedocs.io/en/latest/tutorials/act_max.html)) on high performance computers, use the `activation_maximisation.py`, and `run_act_max_hpc.sh` scripts.
