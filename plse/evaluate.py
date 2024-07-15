import argparse
import logging
import numpy as np
import os
import keras
from plse.data import DataLoader
from plse.utils import verify_output


def evaluate_from_saved_model(
        # Input model and files to be evaluated
        input_files,
        input_model="model.keras",
        # Save settings
        output_file="output.npy",
        overwrite=False,
    ):

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    logging.info("Loading data...")
    dataloader = DataLoader(input_files)
    waveforms = dataloader.load_waveforms()
    encoded_npes = dataloader.load_encoded_npe()

    # Load network
    logging.info("Loading network "+input_model)
    loaded_model = keras.models.load_model(input_model)

    # Verify the output directory and file before evaluating
    logging.info("Verifying the output location and file...")
    output_dir, filename = os.path.split(output_file)
    _, output_file = verify_output(output_dir=output_dir, filename=filename, overwrite=overwrite)

    # Evaluate
    output = loaded_model(waveforms).numpy()

    # Save output
    with open(output_file,'wb') as f:
        np.save(f, output)


def main():
    parser = argparse.ArgumentParser(description='Train PLSECounter model.')
    parser.add_argument('input_files', nargs='+', help='Input files containing the waveforms to be evaluated.')
    parser.add_argument('-m', '--input-model', help='Input model to use for evaluation.',
                        default='model.keras')
    parser.add_argument('-o', '--output-file', help='File to save the evaluated output.',
                        default='output.npy')
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='Force an overwrite of the output.')
    args = parser.parse_args()

    evaluate_from_saved_model(
        # Training files
        input_files = args.input_files,
        input_model = args.input_model,
        # Save settings
        output_file = args.output_file,
        overwrite = args.force_overwrite,
    )

if __name__ == '__main__':
    main()
