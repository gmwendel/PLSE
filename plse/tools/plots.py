import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import argparse

from plse.data import DataLoader

def make_plots(input_files, input_network, output_plots, interactive=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Loading data...")
    dataloader = DataLoader(input_files)
    waveforms = dataloader.load_waveforms()
    encoded_npe = dataloader.load_encoded_npe()

    logging.info("Loading Network "+input_network)
    plse_counter = tf.keras.models.load_model(input_network)

    logging.info("Generating predictions...")
    output = plse_counter(waveforms).numpy()
    con_mat = tf.math.confusion_matrix(labels=encoded_npe.argmax(axis=1),
                                       predictions=output.argmax(axis=1)).numpy()
    print("Confusion Matrix:")
    print(con_mat)

    #assume largest number of p.e. predicted is an overflow bin
    maxpe=output.shape[1]
    labels = [str(i) if i < maxpe-1 else "> "+str(maxpe-2) for i in range(maxpe)]

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=labels, columns=labels)
    mat_df = pd.DataFrame(con_mat, index=labels, columns=labels)
    print("Normalized Confusion Matrix:")
    print(con_mat_norm)

    plt.rc('font', size=30)  # controls default text size
    plt.rc('axes', titlesize=30)  # fontsize of the title
    plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=30)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend
    figure = plt.figure(figsize=(24, 18))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, cbar_kws={'label': 'Fraction of events classified along row'})
    plt.tight_layout()
    plt.ylabel('True # of photons')
    plt.xlabel('Predicted # of photons')
    plt.title('"Confusion Matrix" Normalized Along Rows')
    plt.savefig(output_plots+"_rownorm.png", format="png", bbox_inches='tight')
    if interactive:
        plt.show()
    plt.close()

    plt.rc('font', size=30)  # controls default text size
    plt.rc('axes', titlesize=30)  # fontsize of the title
    plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=30)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=30)  # fontsize of the legend
    figure = plt.figure(figsize=(24, 18))
    sns.heatmap(np.around(mat_df / len(output), 3), annot=True, cmap=plt.cm.Blues, cbar_kws={'label': 'Fraction of Events'})
    plt.tight_layout()
    plt.title('Confusion Matrix Normalized Across All Events')
    plt.ylabel('True # of photons')
    plt.xlabel('Predicted # of photons')
    plt.savefig(output_plots+"_norm.png", format="png", bbox_inches='tight')
    if interactive:
        plt.show()
    plt.close()

def plot_single(input_files, input_network, output_plots, event,interactive=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Loading data...")
    dataloader = DataLoader(input_files)
    waveforms = dataloader.load_waveforms()
    waveforms=np.array([waveforms[event]])
    print(waveforms.shape)
    encoded_npe = dataloader.load_encoded_npe()[event]

    logging.info("Loading Network "+input_network)
    plse_counter = tf.keras.models.load_model(input_network)

    logging.info("Generating predictions...")
    output = plse_counter(waveforms).numpy()
    print(output)
    print(encoded_npe)
    print(waveforms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files',
                        help='Type = String; Input locations of training set files, e.g. $PWD/{1..16}.root', nargs="+",
                        required=True)
    parser.add_argument('-n', '--input_network',
                        help='Type = String; Input locations of network, e.g. $PWD/mynetwork',
                        required=True)
    parser.add_argument('-o', '--output_plots',
                        help='Type = String; Output location for plots, e.g. myplots',
                        required=True)
    args = parser.parse_args()

    make_plots(args.input_files, args.input_network, args.output_plots)
