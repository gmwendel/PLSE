import argparse
import logging
import os
import tensorflow as tf
import keras
from plse.data import DataLoader, DataGenerator
from plse.models import PLSECounter


def train_counter(input_files, network_output, save_history=False, use_multiprocessing=False):
    assert not use_multiprocessing, "`use_multiprocessing` is not currently supported."
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Loading data...")
    dataloader = DataLoader(input_files)
    waveforms = dataloader.load_waveforms()
    encoded_npes = dataloader.load_encoded_npe()

    # Take 1/10 total data and make it validation
    splits = int(len(waveforms) / 10)
    train_dataset = DataGenerator(waveforms[:-splits], encoded_npes[:-splits], -20, 30 + 1)
    validation_dataset = DataGenerator(waveforms[-splits:], encoded_npes[-splits:], -20, 30 + 1)

    logging.info("Building and compiling the model...")

    plse_counter = PLSECounter(waveforms.shape, encoded_npes.shape)
    plse_counter.compile_model()

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)]

    if save_history:
        logging.info("Initializing checkpoints and TensorBoard logs...")
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                os.path.join(network_output, 'resources', 'checkpoints_{epoch:02d}'),
                save_freq='epoch'))
        callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(network_output, 'resources', 'logs'),
                                                     histogram_freq=1))

    logging.info("Start training...")
    history = plse_counter.fit(train_dataset,
                               validation_data=validation_dataset,
                               epochs=1,#000,
                               callbacks=callbacks,
                               )

    plse_counter.save(network_output+"/model.keras")


def main():
    parser = argparse.ArgumentParser(description='Train PLSECounter model.')
    parser.add_argument('input_files', nargs='+', help='Input files containing the waveforms and npe data.')
    parser.add_argument('-n', '--network_output', help='Path to the directory to save the trained model and logs.',
                        default='plse_counter')
    parser.add_argument('--save_history', action='store_true', help='Save training history and checkpoints.')
    parser.add_argument('--use_multiprocessing', action='store_true', default=False,
                        help='Use multiprocessing for data loading.')
    args = parser.parse_args()

    train_counter(args.input_files, args.network_output, args.save_history, args.use_multiprocessing)

if __name__ == '__main__':
    main()
