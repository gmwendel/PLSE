import numpy as np
import os
import warnings
from keras.utils import Sequence


class DataLoader:
    def __init__(self, input_files, npe_cut=10):
        """
        DataLoader constructor.

        Parameters:
            input_files (list of str): Input files to be read containing the waveforms and npe data.
            npe_cut (int): The threshold for the one-hot encoding cutoff.
        """
        self.input_files = self._check_input_files(input_files)
        self.waveforms = None
        self.encoded_npe = None
        self.npe_cut = npe_cut

    def _check_input_files(self, input_files):
        """
        Check if the input files exist.

        Parameters:
            input_files (list of str): Input files to be read containing the waveforms and npe data.

        Returns:
            list of str: Valid input files
        """
        valid_files = []
        for file in input_files:
            if os.path.exists(file):
                valid_files.append(file)
            else:
                warnings.warn(f"File not found: {file}")

        if not valid_files:
            raise ValueError("No valid input files provided.")

        return valid_files

    def load_waveforms(self):
        """
        Load the waveform data from multiple file sources.

        Returns:
            np.ndarray: The 2D waveform data.
        """
        waveforms_list = []

        for file in self.input_files:
            with np.load(file) as data:
                waveforms_list.append(data['waveforms'])

        waveforms = np.concatenate(waveforms_list)
        return waveforms

    def load_event_data(self):
        """
        Load the event data from multiple file sources.

        Returns:
            np.ndarray: The 1D event data.
        """
        event_data_list = []

        for file in self.input_files:
            with np.load(file) as data:
                event_data_list.append(data['eventid'])

        evt_data = np.concatenate(event_data_list)
        return evt_data

    def load_npe(self):
        """
        Load the npe data from multiple file sources.

        Returns:
            np.ndarray: The 1D npe data.
        """
        nphotons_list = []

        for file in self.input_files:
            with np.load(file) as data:
                nphotons_list.append(data['nphotons'])

        nphotons = np.concatenate(nphotons_list)
        return nphotons

    def load_encoded_npe(self):
        """
        Load and preprocess the encoded_npe data.

        Returns:
            np.ndarray: The encoded_npe data.
        """
        nphotons = self.load_npe()
        encoded_npe = DataLoader.one_hot_encode_with_overflow(nphotons, self.npe_cut)
        return encoded_npe

    @staticmethod
    def one_hot_encode_with_overflow(n, n_max):
        """
        One-hot encode a 1D numpy array with an overflow bin at n_max.

        Parameters:
            n (np.ndarray): The 1D numpy array of integers to be one-hot encoded.
            n_max (int): The maximum value for the one-hot encoding (inclusive).

        Returns:
            np.ndarray: One-hot encoded matrix.
        """
        num_bins = n_max + 2  # Including the overflow bin
        encoded_matrix = np.zeros((len(n), num_bins))
        idx_within_range = (n <= n_max)
        idx_overflow = (n > n_max)
        encoded_matrix[np.arange(len(n))[idx_within_range], n[idx_within_range]] = 1
        encoded_matrix[np.arange(len(n))[idx_overflow], -1] = 1  # Overflow bin
        return encoded_matrix.astype(np.float32)


class DataGenerator(Sequence):
    def __init__(self, x, y, batch_size=10000, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = self.x[indexes]
        batch_y = self.y[indexes]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
