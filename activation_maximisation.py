import argparse
import numpy as np
import pandas as pd
import scipy as sp
import torch
import connectome_interpreter as coin


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The script:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inprop = sp.sparse.load_npz(args.inprop_path)
    meta = pd.read_csv(args.meta_path, index_col=0)
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
    # sp.sparse.save_npz('C:\\Users\\44745\\projects\\interpret_connectome\\data\\adult_inprop.npz', inprop)

    # add negative connections
    idx_to_sign = dict(zip(meta.idx, meta.sign))
    inprop_dense = inprop.toarray()
    neg_indices = [idx for idx, val in idx_to_sign.items() if val == -1]
    inprop_dense[neg_indices] = -inprop_dense[neg_indices]

    inprop_tensor = torch.from_numpy(inprop_dense).t().to(device)

    # make model
    sensory_indices = list(
        set(meta.idx[meta.super_class.isin(['sensory', 'visual_projection', 'ascending'])]))
    ml_model = coin.activation_maximisation.MultilayeredNetwork(
        inprop_tensor, sensory_indices, threshold=0, tanh_steepness=5, num_layers=args.num_layers).to(device)

    def regularisation(tensor):
        return torch.norm(tensor, 1)

    target_index_dic = {i: [args.target_index] for i in range(args.num_layers)}

    opt_in, out, act_loss, out_reg_loss, in_reg_los, snapshots = coin.activation_maximisation.activation_maximisation(ml_model,
                                                                                                                      target_index_dic,
                                                                                                                      num_iterations=60, learning_rate=0.4,
                                                                                                                      in_regularisation_lambda=3e-4, custom_in_regularisation=regularisation,
                                                                                                                      out_regularisation_lambda=0.1,
                                                                                                                      device=device,
                                                                                                                      stopping_threshold=1e-6,
                                                                                                                      wandb=False)

    # save the optimised input
    np.save(args.optimised_input_path + '/' +
            'opt_in_' + args.job_id + '.npy', opt_in)
    # save the output
    np.save(args.output_path + '/out_' + args.job_id + '.npy', out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some inputs.')
    # Define expected arguments
    parser.add_argument('--inprop_path', type=str, required=True,
                        help='Path to the connectivity (input proportion) matrix')
    parser.add_argument('--meta_path', type=str, required=True,
                        help='Path to the meta data')
    parser.add_argument('--target_index', type=int, required=True,
                        help='Index of the target cell type to maximise the activation for')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='Number of layers in the network')
    parser.add_argument('--optimised_input_path', type=str, required=True,
                        help='Path to store the optimised input')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to store the output after optimisation')
    parser.add_argument('--job_id', type=str, required=True,
                        help='Job ID to be used in the result name')
    # Add more arguments as needed

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
