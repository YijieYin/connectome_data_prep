from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import scipy as sp
import torch
import connectome_interpreter as coin

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data 
    inprop = sp.sparse.load_npz(args.inprop_path)
    meta = pd.read_csv(args.meta_path, index_col=0)
    # load data on bumps 
    bumps = pd.read_csv('data/PENandEPG_wedges.csv')
    if meta.idx.isna().any():
        print('Some cell types are not taken into account: there are NAs in the indices in meta.')
        meta = meta[~meta.idx.isna()]
    meta.idx = meta.idx.astype(int)

    # remove connections from KCs and DANs to KCs, since they are axo-axonic
    inprop = coin.utils.modify_coo_matrix(inprop, input_idx=meta.idx[meta.cell_class.isin(['Kenyon_Cell', 'DAN'])].unique(),
                                          output_idx=meta.idx[meta.cell_class == 'Kenyon_Cell'].unique(), value=0)
    inprop = coin.utils.modify_coo_matrix(inprop,
                                    set(meta.idx[meta.cell_class == 'Kenyon_Cell']),
                                    set(meta.idx[meta.cell_class == 'DAN']),
                                    0)                                      

    # add negative connections
    idx_to_sign = dict(zip(meta.idx, meta.sign))
    inprop_dense = inprop.toarray()
    neg_indices = [idx for idx, val in idx_to_sign.items() if val == -1]
    inprop_dense[neg_indices] = -inprop_dense[neg_indices]

    # make model
    # sorting sensory_indices is important because it makes the ordering of indices reproducible down the line
    # set() doesn't preserve the order
    # sensory_indices = sorted(list(
    #     set(meta.idx[meta.super_class.isin(['sensory', 'visual_projection', 'ascending'])])))
    # take olfactory neurons ony 
    sensory_indices = sorted(list(
        set(meta.idx[meta.cell_class == 'olfactory'])
    ))

    def regularisation(tensor):
        return torch.norm(tensor, 1)
    
    # LTD/LTP -------------
    # ltd_neurons = {'pre':meta.idx[meta.cell_sub_class == 'ring neuron'].tolist(), 
    #                 'post': meta.idx[meta.cell_type == 'EPG'].tolist()}
    # pre, post = np.meshgrid(ltd_neurons['pre'], ltd_neurons['post'])
    # # Flatten the arrays and create a DataFrame
    # ltd_df = pd.DataFrame({'pre': pre.ravel(), 'post': post.ravel()})

    # moving bump --------
    # epgs = set(meta.idx[meta.cell_type == 'EPG'])

    # bump1 = meta.idx[meta.root_id.astype(str).isin(bumps.root_id[bumps.name.isin(['EPG_L3','EPG_R7'])])].tolist()
    # bump2 = meta.idx[meta.root_id.astype(str).isin(bumps.root_id[bumps.name.isin(['EPG_L2','EPG_R8'])])].tolist() 
    # bump3 = meta.idx[meta.root_id.astype(str).isin(bumps.root_id[bumps.name.isin(['EPG_L1','EPG_R1'])])].tolist()

    # i = 10
    # target_index_dic = dict.fromkeys(range(i), bump1)
    # target_index_dic.update(dict.fromkeys(range(i,i+2,1), bump2))
    # target_index_dic.update(dict.fromkeys(range(i+2,i+4,1), bump3))

    # neurons_to_deactivate = dict.fromkeys(range(i), list(epgs - set(bump1)))
    # neurons_to_deactivate.update(dict.fromkeys(range(i,i+2,1), list(epgs - set(bump2))))
    # neurons_to_deactivate.update(dict.fromkeys(range(i+2,i+4,1), list(epgs - set(bump3))))

    act_losses = []
    for ctype in tqdm(meta.cell_type[meta.cell_type.str.contains('CB.FB')].unique()):
        target_index_dic = dict.fromkeys(range(5,9), 
                                         meta.idx[meta.cell_type == ctype].tolist())
        
        # define the model --------
        ml_model = coin.activation_maximisation.MultilayeredNetwork(
            all_weights = torch.from_numpy(inprop_dense).float().t().to(device), 
            sensory_indices = sensory_indices, threshold=0, tanh_steepness=5, 
            num_layers=max(list(target_index_dic.keys()) 
                        #    + list(neurons_to_deactivate.keys())
                           ) + 1
            # num_layers = 10
            ).to(device)

        torch.cuda.empty_cache()
        opt_in, out, act_loss, out_reg_loss, in_reg_los, snapshots = coin.activation_maximisation.activation_maximisation(ml_model,
                                                                                                                        target_index_dic, 
                                                                                                                        #   neurons_to_deactivate = neurons_to_deactivate, 
                                                                                                                        num_iterations=args.num_iterations, learning_rate=0.4,
                                                                                                                        in_regularisation_lambda=3e-4, custom_in_regularisation=regularisation,
                                                                                                                        out_regularisation_lambda=10,
                                                                                                                        #   ltd_df = ltd_df, ltd_coefficient=0.5,
                                                                                                                        device=device,
                                                                                                                        use_tqdm=False,
                                                                                                                        stopping_threshold=1e-6, report_memory_usage = args.report_memory_usage,
                                                                                                                        wandb=args.wandb)

        # save the optimised input
        np.save(args.optimised_input_path + 'opt_in_' +
                args.array_id + '_' + ctype + '.npy', opt_in)
        # save the output
        np.save(args.output_path + 'out_' + args.array_id + '_' + ctype + '.npy', out)
        # np.save(args.weights_path + args.array_id + '.npy', weights)
        act_losses.append(act_loss[-1])
    
    # save activation loss term 
    act_df = pd.DataFrame({'cell_type': meta.cell_type[meta.cell_type.str.contains('CB.FB')].unique(), 
                           'activation': act_losses})
    act_df.to_csv(args.optimised_input_path + args.array_id + '_activations.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some inputs.')
    # Define expected arguments
    parser.add_argument('--inprop_path', type=str, required=True,
                        help='Path to the connectivity (input proportion) matrix')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to the meta data')
    parser.add_argument('--num_iterations', type=int, required=True,
                        help='Number of iterations in activation maximisation')
    parser.add_argument('--optimised_input_path', type=str, required=True,
                        help='Path to store the optimised input')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to store the output after optimisation')
    parser.add_argument('--weights_path', type=str, required=False, 
                        help='Path to save the updated weights to.')
    parser.add_argument('--array_id', type=str, required=True,
                        help='Array ID to be used in the result name')
    parser.add_argument('--report_memory_usage', action='store_true', required=False, default=False, 
                        help='Add this flag to report GPU memory usage before, during and after optimisation.')
    parser.add_argument('--wandb', action='store_true', required=False, default=False,
                        help='Add this flag if you want to use Weights & Biases')
    # Add more arguments as needed

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
