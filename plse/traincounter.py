import argparse
import logging
import os
import tensorflow as tf
import keras
import yaml
from plse.data import DataLoader, DataGenerator, NtupleDataLoader
from plse.models import PLSECounter
import numpy as np

def train_counter(
        # Training files
        input_files,
        # Save settings
        output_dir,
        model_filename="model.keras",
        overwrite=False,
        save_history=False,
        export_tf=False,
        # Model settings
        mode='counter',
        # Training settings
        max_epochs=50,
        early_stopping_patience=5,
        learning_rate_patience=3,
        use_multiprocessing=False,
):
    assert not use_multiprocessing, "`use_multiprocessing` is not currently supported."
    assert learning_rate_patience < early_stopping_patience, "Learning rate patience should be smaller than earlier stopping patience."

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check file extensions and choose appropriate DataLoader
    extensions = set()
    for file in input_files:
        _, ext = os.path.splitext(file)
        extensions.add(ext.lower())

    if len(extensions) > 1:
        raise ValueError("All input files must have the same extension.")
    elif len(extensions) == 0:
        raise ValueError("No input files provided.")
    else:
        ext = extensions.pop()
        if ext == '.root':
            use_ntuple_loader = True
        elif ext == '.npz':
            use_ntuple_loader = False
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are .root and .npz")

    # Load data
    # Load data
    logging.info("Loading data...")
    if use_ntuple_loader:
        dataloader = NtupleDataLoader(input_files, npe_cut=10)
        waveforms, encoded_npes, pe_times, pmt_types = dataloader.load_good_data()
    else:
        dataloader = DataLoader(input_files, npe_cut=10)
        waveforms, encoded_npes, pe_times = dataloader.load_good_data()
        pmt_types = np.zeros(len(waveforms), dtype=np.float32)  # Placeholder if pmt_types not available

    # Define true network output
    true_output = encoded_npes if mode == 'counter' else pe_times[:, 0:1] / 100.

    # Create datasets
    splits = int(len(waveforms) / 10)
    augment_data = True if mode == 'counter' else False
    train_dataset = DataGenerator(waveforms[:-splits], pmt_types[:-splits], true_output[:-splits],
                                  augment_data=augment_data, max_shift_left=20, max_shift_right=30 + 1)
    validation_dataset = DataGenerator(waveforms[-splits:], pmt_types[-splits:], true_output[-splits:],
                                       augment_data=augment_data, max_shift_left=20, max_shift_right=30 + 1)

    logging.info("Building and compiling the model...")

    plse_counter = PLSECounter(waveforms.shape, true_output.shape, counter=True if mode == 'counter' else False,
                               output_length=None if mode == 'counter' else 1)  # TODO: currently only supports single PE timing
    plse_counter.compile_model()

    # Verify the output directory and file before training
    logging.info("Verifying the output location and file...")
    plse_counter.verify_output_args(output_dir, filename=model_filename, overwrite=overwrite)

    # Save the settings used for training to a yaml file in output_dir
    training_metadata = {
        'max_epochs': max_epochs,
        'early_stopping_patience': early_stopping_patience,
        'learning_rate_patience': learning_rate_patience,
    }
    training_metadata_filename = os.path.join(output_dir, 'training_metadata.yml')
    with open(training_metadata_filename, 'w') as f:
        yaml.dump(training_metadata, f)

    # Set up callbacks
    callbacks = []

    # Early stopping callback
    earlystopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    callbacks.append(earlystopping_callback)

    # Learning rate Reduce on Plateau
    learningrate_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=learning_rate_patience)
    callbacks.append(learningrate_callback)

    if save_history:
        logging.info("Initializing checkpoints and TensorBoard logs...")
        # Make subdir for saving history
        resources_dir = os.path.join(plse_counter.output_args['output_dir'], 'callbacks')
        # Checkpoint callback
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(resources_dir, 'checkpoints_epoch-{epoch:02d}.keras'),
            save_freq='epoch',
        )
        callbacks.append(checkpoint_callback)
        # TensorBoard log callback
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=os.path.join(resources_dir, 'tensorboard_logs'),
            histogram_freq=1,
        )
        callbacks.append(tensorboard_callback)

    logging.info("Start training...")
    history = plse_counter.fit(train_dataset,
                               validation_data=validation_dataset,
                               epochs=max_epochs,
                               callbacks=callbacks,
                               )

    # Save keras model
    plse_counter.save()
    # Export TensorFlow model
    if export_tf:
        plse_counter.export_tf()


def main():
    parser = argparse.ArgumentParser(description='Train PLSECounter model.')
    parser.add_argument('input_files', nargs='+', help='Input files containing the waveforms, npe data, and times.')
    parser.add_argument('-o', '--output-dir', help='Path to the directory to save the trained model and logs.',
                        default='./plse_counter')
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='Force an overwrite of the output.')
    parser.add_argument('--export-tf', action='store_true', help='Export TensorFlow saved model.')
    parser.add_argument('--save_history', action='store_true', help='Save training history and checkpoints.')
    parser.add_argument('--mode', default='counter', help='Whether to run in counter mode or timing mode.')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of epochs allowed during training.')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Number of epochs with no improvement in val loss in order to stop training.')
    parser.add_argument('--learning-rate-patience', type=int, default=3,
                        help='Number of epochs with no improvement in val loss in order to reduce learning rate.')
    parser.add_argument('--use_multiprocessing', action='store_true', default=False,
                        help='Use multiprocessing for data loading.')
    args = parser.parse_args()

    train_counter(
        # Training files
        input_files=args.input_files,
        # Save settings
        output_dir=args.output_dir,
        overwrite=args.force_overwrite,
        save_history=args.save_history,
        export_tf=args.export_tf,
        # Model settings
        mode=args.mode,
        # Training settings
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate_patience=args.learning_rate_patience,
        use_multiprocessing=args.use_multiprocessing,
    )


if __name__ == '__main__':
    main()
