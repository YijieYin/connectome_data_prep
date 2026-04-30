This repo contains the pre-processed publicly-available connectomics data (in `data` folder), and the code that processed them. For example usage/analysis, see [this repo](https://github.com/YijieYin/connectome_interpreter). For any questions/requests/thoughts/comments, please feel free to reach out to me at yy432[at]cam.ac.uk :). 

## Data 
### connectome data 
All pre-processed data are in the `data` folder, typically composed of a `scipy.sparse.matrix` and a meta `.csv` file. 
- Adult fruit fly (*Drosophila melanogaster*):
  - maleCNS obtained from [Berg et al. 2025](https://www.biorxiv.org/content/10.1101/2025.10.09.680999v2) and [Nern et al. 2024](https://www.biorxiv.org/content/10.1101/2024.04.16.589741v2), also publicly available on [neuprint](https://male-cns.janelia.org) and [codex](https://codex.flywire.ai/?dataset=mcns).
    - axon-dendrite-only connectivity is available, generated using the code in `mcns_ad_split`. 
  - BANC (brain and nerve cord) from [Bates et al. 2025](https://www.biorxiv.org/content/10.1101/2025.07.31.667571v1), also available on [codex](https://codex.flywire.ai/?dataset=banc). 
  - Full Adult Fly Brain (FAFB) / FlyWire from [Dorkenwald et al. 2024](https://www.nature.com/articles/s41586-024-07558-y), [Schlegel et al. 2024](https://www.nature.com/articles/s41586-024-07686-5), and [Matsliah et al. 2024](https://www.nature.com/articles/s41586-024-07981-1). Connectivity information from [Buhmann et al. 2021](https://www.nature.com/articles/s41592-021-01183-7) and [Yu et al. 2025](https://www.biorxiv.org/content/10.1101/2025.07.11.664377v1), and neurotransmitter data from [Eckstein et al. 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6). By using the connectivity information, you agree to follow the [FlyWire citation guidelines and principles](https://codex.flywire.ai/api/download). To explore the dataset in detail in a cool interface, you can go here: [https://tinyurl.com/flywire783](https://tinyurl.com/flywire783). Also available on [codex](https://codex.flywire.ai/?dataset=fafb).
    - axon-dendrite-only connectivity is available, generated using the code in `fafb_ad_split`. 
  - Hemibrain from [Scheffer et al. 2020](https://doi.org/10.7554/eLife.57443), also available on [neuprint](https://neuprint.janelia.org/?dataset=hemibrain%3Av1.2.1&qt=findneurons), and 
  - MANC (male adult nerve cord) from [Cheong et al. 2025](https://www.biorxiv.org/content/10.1101/2023.06.07.543976v3), [Marin et al. 2024](https://www.biorxiv.org/content/10.1101/2023.06.05.543407v2), and [Takemura et al. 2024](https://elifesciences.org/reviewed-preprints/97769). Also available on [neuprint](https://neuprint.janelia.org/?dataset=manc:v1.2.1&qt=findneurons). 

- Larva fruit fly (*Drosophila melanogaster*): the processed version of the larval connectivity is `larva_inprop.npz` (using axon-dendrite-only connections). Both the cell type annotaitons and the connectivity came from [Winding et al. 2023](https://www.science.org/doi/10.1126/science.add9330). You can also e.g. visualise the neurons in 3D in [catmaid](https://catmaid.virtualflybrain.org/).

Generally, `inprop` stands for input proportion (where the connectivity is normalised by the total amount of input for the recipient neuron / cell type), `outprop` stands for output proportion, `syncount` stands for synapse count. `cb` stands for central brain, `optic` stands for optic lobe. `ad` stands for axon-dendrite connectivity. 

### experimental data 
This repository also collates published experimental data (sometimes scrapped with [WebPlotDigitizer](https://automeris.io/)), often linking between sensory space (e.g. odours) to neuron activation space (e.g. sensory neuron activations), i.e. what is the neuron activation response upon the presentation of some stimulus. The datasets are also in `data`. 
- [Münch & Galizia 2016](https://www.nature.com/articles/srep21841): DoOR 2.0 - Comprehensive Mapping of Drosophila melanogaster Odorant Responses
- [Badel et al. 2016](https://www.cell.com/neuron/abstract/S0896-6273(16)30201-X): Decoding of Context-Dependent Olfactory Behavior in Drosophila
- [Bhandawat et al. 2007](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2838615/): Sensory Processing in the Drosophila Antennal Lobe Increases the Reliability and Separability of Ensemble Odor Representations 
- [Dolan et al. 2018](https://www.cell.com/neuron/fulltext/S0896-6273(18)30742-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0896627318307426%3Fshowall%3Dtrue): Communication from Learned to Innate Olfactory Processing Centers Is Required for Memory Retrieval in Drosophila
- [Dweck et al. 2018](https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6): The Olfactory Logic behind Fruit Odor Preferences in Larval and Adult Drosophila
- [Hallem & Carlson 2006](https://www.cell.com/cell/abstract/S0092-8674(06)00363-1): Coding of Odors by a Receptor Repertoire
- [Liu et al. 2022](https://www.cell.com/current-biology/fulltext/S0960-9822(21)01642-0): Connectomic features underlying diverse synaptic connection strengths and subcellular computation
- [Semmelhack & Wang 2009](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2702439/): Select Drosophila glomeruli mediate innate olfactory attraction and aversion
- [Frechter et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6550879/): Functional and anatomical specificity in a higher olfactory centre

The code for retrieving / tidying up the data is also in the root repository / in their respective folders. 

## Data processing 
The code used to generate the sparse matrices and metadata files above are in respective folders, including info on where the connectome data is downloaded from, e.g. `FAFB` for code in generating sparse matrices based on raw data on edgelist of synapse count, and joining multiple cell-type-like columns. 

The code for generating axon-dendrite split is in folder `*_ad_split`, where neurons are split using the flow centrality method in Fig 7 in [Schneider-Mizell et al. 2016](https://elifesciences.org/articles/12059). Note that about 20k neurons are left not split, due to low segregation index. This thus generates 9 edgelists: aa (axo-axonic), ad (axo-dendritic), ab (axon to not-split neurons, i.e. both axon and dendrite), da, dd, db, ba, bd, bb (too big to share here, available on request). The axo-dendritic connectome is made using ad, ab, bd, bb. 

## Analysis 
This respository also contains code that runs some of the analyses included in [Yin et al. 2025](https://www.biorxiv.org/content/10.1101/2025.09.29.679410v2), including
- `matmul_benchmark`: bench-marking the speed of [`compress_paths()` function](https://connectome-interpreter.readthedocs.io/en/latest/modules/compress_paths.html#connectome_interpreter.compress_paths.compress_paths) for [sparse matrix powers](https://connectome-interpreter.readthedocs.io/en/latest/tutorials/matmul.html) in [Connectome Interpreter](https://github.com/YijieYin/connectome_interpreter).
- `pathfinding_benchmark`: bench-marking the speed of [`find_paths_of_length()` function](https://connectome-interpreter.readthedocs.io/en/latest/modules/path_finding.html#connectome_interpreter.path_finding.find_paths_of_length) for [path-finding](https://connectome-interpreter.readthedocs.io/en/latest/tutorials/path_finding.html).
- `path_effconn_benchmark`: bench-marking on connectivity density and magnitude of effective connectivity across path lengths.
- `quantify_recurrence`: quantifying the proportion of cells with self-loops (without additional loops) for (axon-dendrite split) (excitation-only) connectome, and the effective excitation/inhibition.
- `eonly_pathfinding`: connectivity density for excitaiton-only / excitation-only connections. 

## Data sharing 
The results are shared here whenever <100MB. Bigger results are available on request. 
