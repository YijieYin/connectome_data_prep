import argparse
import os
import numpy as np


def main(args):
    print('Concatenating npy files...')
    files = os.listdir(args.folder_to_concat)
    all_data = []
    for file in files:
        if file.endswith('.npy'):
            # this is an array of shape (num_neurons, num_timepoints)
            data = np.load(os.path.join(args.folder_to_concat, file))
            data_v = data.flatten()
            all_data.append(data_v)
    all_data = np.vstack(all_data)
    np.save(args.output_path, all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some inputs.')
    # Define expected arguments
    parser.add_argument('--folder_to_concat', type=str, required=True,
                        help='Path to folder containing the files to concatenate')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output file')
    # Add more arguments as needed

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
