import numpy as np
##import tensorflow as tf
import argparse

import matplotlib.pyplot as plt

from plse.data import DataLoader

def plot_true_hit_times(input_files, output_file):

    # Load truth data
    dataloader = DataLoader(input_files)
    encoded_npe = dataloader.load_encoded_npe()
    print(encoded_npe)
    times = dataloader.load_times()
    print('All hit times:')
    print(times)
    print('First hit times:')
    print(times[:,0])
    print('Earliest and latest first hit times:')
    print(min(times[:,0]),max(times[:,0]))

    min_t = min(times[:,0])
    max_t = max(times[:,0])
    max_t = 250 if not np.isfinite(max_t) else max_t
    plt.hist(times[:,0],bins=np.linspace(min_t,max_t,101))
    plt.yscale('log')
    plt.xlabel('True hit times')
    plt.savefig(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files',
                        help='Type = String; Input locations of training set files, e.g. $PWD/{1..16}.root', nargs="+",
                        required=True)
    parser.add_argument('-o', '--output_file',
                        help='Type = String; Output file name to save plot',
                        required=False, default="hit_times.png")
    args = parser.parse_args()

    plot_true_hit_times(args.input_files, args.output_file)
