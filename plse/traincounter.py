import argparse
import logging
import os
import tensorflow as tf
import keras
from plse.data import DataLoader, DataGenerator
from plse.models import PLSECounter


def train_counter(
        # Training files
        input_files,
        # Save settings
        output_dir,
        model_filename="model.keras",
        overwrite=False,
        save_history=False,
        export_tf=False,
        # Training settings
        use_multiprocessing=False,
    ):

    assert not use_multiprocessing, "`use_multiprocessing` is not currently supported."

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
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

    # Verify the output directory and file before training
    logging.info("Verifying the output location and file...")
    plse_counter.verify_output_args(output_dir, filename=model_filename, overwrite=overwrite)

    # Set up callbacks
    callbacks = []

    # Early stopping callback
    earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
    callbacks.append(earlystopping_callback)

    if save_history:
        logging.info("Initializing checkpoints and TensorBoard logs...")
        # Make subdir for saving history
        resources_dir = os.path.join(plse_counter.output_args['output_dir'], 'callbacks')
        # Checkpoint callback
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(resources_dir, 'checkpoints_epoch-{epoch:02d}.keras'),
            save_freq = 'epoch',
        )
        callbacks.append(checkpoint_callback)
        # TensorBoard log callback
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir = os.path.join(resources_dir, 'tensorboard_logs'),
            histogram_freq = 1,
        )
        callbacks.append(tensorboard_callback)

    logging.info("Start training...")
    history = plse_counter.fit(train_dataset,
                               validation_data=validation_dataset,
                               epochs=1,
                               callbacks=callbacks,
                               )

    # Save keras model
    plse_counter.save()
    # Export TensorFlow model
    if export_tf:
        plse_counter.export_tf()

def main():
    parser = argparse.ArgumentParser(description='Train PLSECounter model.')
    parser.add_argument('input_files', nargs='+', help='Input files containing the waveforms and npe data.')
    parser.add_argument('-o', '--output-dir', help='Path to the directory to save the trained model and logs.',
                        default='./plse_counter')
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='Force an overwrite of the output.')
    parser.add_argument('--export-tf', action='store_true', help='Export TensorFlow saved model.')
    parser.add_argument('--save_history', action='store_true', help='Save training history and checkpoints.')
    parser.add_argument('--use_multiprocessing', action='store_true', default=False,
                        help='Use multiprocessing for data loading.')
    args = parser.parse_args()

    train_counter(
        # Training files
        input_files = args.input_files,
        # Save settings
        output_dir = args.output_dir,
        overwrite = args.force_overwrite,
        save_history = args.save_history,
        export_tf = args.export_tf,
        # Training settings
        use_multiprocessing = args.use_multiprocessing,
    )

if __name__ == '__main__':
    main()
