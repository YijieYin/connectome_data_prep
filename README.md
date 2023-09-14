# interpret_connectome
This repo has the code and some of the output for making use of connectomics data. The `data` folder has data from papers that I have used. 

## Code 
- The code on `activation maximisation` is in the `activation maximisation.ipynb` file.
- Most other plots are generated in `ad_exploration.ipynb`. It is so named because I am using the axo-dendritic connections of the larval connectome only from [Winding et al. 2023](https://www.science.org/doi/10.1126/science.add9330). 

## Interactive plots 
- [This](https://yijieyin.github.io/interpret_connectome/htmls/interactive_umap_from_senses_by_type.html) is a UMAP of neurons' input profile grouped by sensory modality (e.g. a neuron receives 20% of its input from olfaction). There are `number_of_modalities` numbers for each neuron. 
- [This](https://yijieyin.github.io/interpret_connectome/htmls/interactive_umap_all_input_by_type.html) is a UMAP of neurons' input profile per type (e.g. type a receives 2% of its input from type b). There are `number_of_types` numbers for each neuron. 
- [**This**](https://yijieyin.github.io/interpret_connectome/htmls/interactive_umap_all_input_by_type_with_bars.html) (recommended) is a UMAP of neurons' input profile per type + bar plots of contributions from sensory neurons. 
