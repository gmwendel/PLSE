import awkward as ak
import glob
import numpy as np
import os
import uproot
import warnings
from keras.utils import Sequence


def DataLoader(input_files, npe_cut=10):
    '''
    General data loader that will choose between Npz or Ntuple depending on the input file extension.
    '''

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
    if use_ntuple_loader:
        dataloader = NtupleDataLoader(input_files, npe_cut=10)
    else:
        dataloader = NpzDataLoader(input_files, npe_cut=10)
    return dataloader


class NpzDataLoader:
    def __init__(self, input_files, npe_cut=10):
        """
        NpzDataLoader constructor.

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

    def load_times(self):
        """
        Load the hit time data from multiple file sources.

        Returns:
            np.ndarray: The 2D time data.
        """
        times_list = []

        for file in self.input_files:
            with np.load(file) as data:
                times_list.append(data['hittimes'])

        times = np.concatenate(times_list)
        return times

    def load_encoded_npe(self):
        """
        Load and preprocess the encoded_npe data.

        Returns:
            np.ndarray: The encoded_npe data.
        """
        nphotons = self.load_npe()
        encoded_npe = DataLoader.one_hot_encode_with_overflow(nphotons, self.npe_cut)
        return encoded_npe

    def load_good_data(self):
        """
        Load the waveforms, encoded_npe, and pe_times.
        Exclude waveforms where a true photon time is not finite, while printing a warning.

        Returns:
            np.ndarray, np.ndarray, np.ndarray: The waveforms, encoded_npe, and pe_times data.
        """
        waveforms = self.load_waveforms()
        encoded_npes = self.load_encoded_npe()
        pe_times = self.load_times()

        # Make a mask excluding infs and nans
        good_event_mask = np.all(np.isfinite(pe_times), axis=1)
        print(len(good_event_mask))

        # Check for bad pe times
        if np.sum(~good_event_mask) > 0:
            print('\n\n ---> WARNING!! Bad PE times present in the files!!! %d events will be removed.\n\n' % np.sum(
                ~good_event_mask))

        return waveforms[good_event_mask], encoded_npes[good_event_mask], pe_times[good_event_mask]

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


class NtupleDataLoader:
    """
    A class to load and manage waveform and meta data from multiple ROOT files.

    This class accepts multiple file paths (including glob patterns), loads the meta data ensuring
    consistency across all files, and provides methods to load large datasets on demand as if they
    came from a single file.

    Attributes:
        digitizer_sample_rate (np.float32): Consistent digitizer sample rate extracted from meta data.
        digitizer_bit_to_mV (np.float32): Consistent digitizer bit to mV conversion factor from meta data.
        pmtid_to_type (np.ndarray of np.int32): Consistent PMT ID to type mapping from meta data.
    """

    def __init__(self, input_files, npe_cut=10):
        """
        Initialize the NtupleDataLoader with multiple ROOT files.

        Parameters:
            input_files (list of str): Input files to be read containing the waveforms and npe data.
            npe_cut (int): The threshold for the one-hot encoding cutoff.
        """
        self.input_files = self._check_input_files(input_files)
        self.npe_cut = npe_cut
        self._load_meta_data()

        # Initialize placeholders for data to be loaded on demand
        self._evid = None
        self._waveform_pmtid = None
        self._inWindowPulseTimes = None
        self._sorting_indices_list = None
        self._waveforms = None
        self._nphotons = None  # Added placeholder for nphotons

    def _check_input_files(self, input_files):
        """
        Check if the input files exist and expand glob patterns.

        Parameters:
            input_files (list of str): Input files to be read containing the waveforms and npe data.

        Returns:
            list of str: Valid input files
        """
        # Ensure input_files is a list
        if isinstance(input_files, str):
            input_files = [input_files]

        # Expand glob patterns and get the full list of files
        file_list = list(set(file for path in input_files for file in glob.glob(path)))

        # Check that there is at least one file
        if not file_list:
            raise ValueError("No files found matching the provided input files.")

        # Check if files exist
        valid_files = []
        for file in file_list:
            if os.path.exists(file):
                valid_files.append(file)
            else:
                warnings.warn(f"File not found: {file}")

        if not valid_files:
            raise ValueError("No valid input files provided.")

        return valid_files

    def _load_meta_data(self):
        """
        Load meta data from all files and check consistency.
        """

        def read_meta(file_path):
            with uproot.open(file_path) as file:
                meta_tree = file["meta"]
                digitizer_sample_rate = np.float32(meta_tree["digitizerSampleRate_GHz"].array(library="np")[0])
                digitizer_bit_to_mV = np.float32(meta_tree["digitizerResolution_mVPerADC"].array(library="np")[0])
                pmtid_to_type = meta_tree["pmtType"].array(library="np")[0].astype(np.int32)
            return digitizer_sample_rate, digitizer_bit_to_mV, pmtid_to_type

        meta_data = [read_meta(fp) for fp in self.input_files]
        self.digitizer_sample_rate_list, self.digitizer_bit_to_mV_list, self.pmtid_to_type_list = zip(*meta_data)

        # Check consistency of scalar meta data
        if not all(self.digitizer_sample_rate_list[0] == x for x in self.digitizer_sample_rate_list):
            raise ValueError("Inconsistent digitizer sample rates across files.")
        if not all(self.digitizer_bit_to_mV_list[0] == x for x in self.digitizer_bit_to_mV_list):
            raise ValueError("Inconsistent digitizer bit to mV values across files.")
        if not all(np.array_equal(self.pmtid_to_type_list[0], x) for x in self.pmtid_to_type_list):
            raise ValueError("Inconsistent pmtid_to_type across files.")

        # Store the consistent meta data
        self.digitizer_sample_rate = self.digitizer_sample_rate_list[0]  # np.float32
        self.digitizer_bit_to_mV = self.digitizer_bit_to_mV_list[0]  # np.float32
        self.pmtid_to_type = self.pmtid_to_type_list[0]  # np.int32

    def _load_waveforms_tree_array(self, branch_name, dtype):
        """
        Helper function to load and concatenate arrays from the waveforms tree.

        Parameters:
            branch_name (str): Name of the branch to load.
            dtype (np.dtype): Data type to cast the array.

        Returns:
            np.ndarray: Concatenated array from all files.
        """
        arrays = []
        for file_path in self.input_files:
            with uproot.open(file_path) as file:
                waveforms_tree = file["waveforms"]
                data = waveforms_tree[branch_name].array(library="np").astype(dtype)
                arrays.append(data)
        return np.concatenate(arrays)

    def load_event_data(self):
        """
        Load and return evid data from all files.

        Returns:
            np.ndarray of np.int32: Concatenated array of event IDs.
        """
        if self._evid is None:
            self._evid = self._load_waveforms_tree_array("evid", np.int32)
        return self._evid

    def load_pmtid(self):
        if self._waveform_pmtid is None:
            pmtid_arrays = []
            for file_path in self.input_files:
                with uproot.open(file_path) as file:
                    waveforms_tree = file["waveforms"]
                    pmtids = waveforms_tree["waveform_pmtid"].array(library="np").astype(np.int32)
                    pmtid_arrays.append(pmtids)
            self._waveform_pmtid = np.concatenate(pmtid_arrays)
        return self._waveform_pmtid

    def load_pmt_type(self):
        pmtids = self.load_pmtid()
        pmt_types = self.pmtid_to_type[pmtids]
        return pmt_types.astype(np.float32)

    def load_npe(self):
        """
        Calculate and return the number of photons in each waveform.

        Returns:
            np.ndarray of np.int32: Array containing the number of photons per waveform.
        """
        if self._nphotons is None:
            if hasattr(self, '_raw_inWindowPulseTimes'):
                inWindowPulseTimes = self._raw_inWindowPulseTimes
            else:
                # Load raw inWindowPulseTimes
                raw_inWindowPulseTimes = []
                for file_path in self.input_files:
                    with uproot.open(file_path) as file:
                        waveforms_tree = file["waveforms"]
                        inWindowPulseTimes = waveforms_tree["inWindowPulseTimes"].array(library="ak")
                        raw_inWindowPulseTimes.append(inWindowPulseTimes)
                inWindowPulseTimes = ak.concatenate(raw_inWindowPulseTimes)
            # Calculate the number of photons
            nphotons = ak.num(inWindowPulseTimes)
            self._nphotons = ak.to_numpy(nphotons).astype(np.int32)
        return self._nphotons

    def load_times(self, nentries=100):
        """
        Load, sort, pad, and return inWindowPulseTimes data from all files.

        Parameters:
            nentries (int): Desired fixed length for each pulse time array.

        Returns:
            np.ndarray of np.float32: 2D array with sorted and padded in-window pulse times.
        """
        if self._inWindowPulseTimes is None:
            inWindowPulseTimes_list = []
            raw_inWindowPulseTimes = []
            self._sorting_indices_list = []  # Store per file
            for file_path in self.input_files:
                with uproot.open(file_path) as file:
                    waveforms_tree = file["waveforms"]
                    inWindowPulseTimes = waveforms_tree["inWindowPulseTimes"].array(library="ak")
                    raw_inWindowPulseTimes.append(inWindowPulseTimes)
                    # Get sorting indices
                    sorting_indices = ak.argsort(inWindowPulseTimes)
                    # Sort times using the sorting indices
                    inWindowPulseTimes_sorted = inWindowPulseTimes[sorting_indices]
                    lengths = ak.num(inWindowPulseTimes_sorted)
                    too_long = lengths > nentries
                    if ak.any(too_long):
                        num_too_long = ak.sum(too_long)
                        print(
                            f"Warning: {num_too_long} events have more pulse times than nentries ({nentries}). Cutting off extra times.")
                    # Pad or truncate times and sorting indices to nentries
                    inWindowPulseTimes_padded = ak.pad_none(inWindowPulseTimes_sorted, nentries, clip=True)
                    sorting_indices_padded = ak.pad_none(sorting_indices, nentries, clip=True)
                    # Replace None with -999 in times, and with -1 in sorting indices
                    inWindowPulseTimes_filled = ak.fill_none(inWindowPulseTimes_padded, -999)
                    sorting_indices_filled = ak.fill_none(sorting_indices_padded, -1)
                    # Convert times to NumPy array
                    inWindowPulseTimes_np = ak.to_numpy(inWindowPulseTimes_filled).astype(np.float32)
                    inWindowPulseTimes_list.append(inWindowPulseTimes_np)
                    # Store sorting indices as awkward array
                    self._sorting_indices_list.append(sorting_indices_filled)
            # Concatenate all arrays
            self._inWindowPulseTimes = np.concatenate(inWindowPulseTimes_list, axis=0)
            # Also store raw inWindowPulseTimes for load_npe
            self._raw_inWindowPulseTimes = ak.concatenate(raw_inWindowPulseTimes)
        return self._inWindowPulseTimes

    def load_charges(self, nentries=100):
        """
        Load, sort (using the same sorting as times), pad, and return inWindowPulseCharges data from all files.

        Parameters:
            nentries (int): Desired fixed length for each pulse charge array.

        Returns:
            np.ndarray of np.float32: 2D array with sorted and padded in-window pulse charges.
        """
        if self._inWindowPulseCharges is None:
            if not hasattr(self, '_sorting_indices_list') or self._sorting_indices_list is None:
                raise ValueError("Sorting indices are not available. Please run load_times first.")
            inWindowPulseCharges_list = []
            idx_file = 0
            for file_path in self.input_files:
                with uproot.open(file_path) as file:
                    waveforms_tree = file["waveforms"]
                    inWindowPulseCharges = waveforms_tree["inWindowPulseCharges"].array(library="ak")
                    # Pad or truncate charges to nentries
                    inWindowPulseCharges_padded = ak.pad_none(inWindowPulseCharges, nentries, clip=True)
                    # Replace None with -999
                    inWindowPulseCharges_filled = ak.fill_none(inWindowPulseCharges_padded, -999)
                    # Get the corresponding sorting indices
                    sorting_indices_filled = self._sorting_indices_list[idx_file]
                    # Replace -1 with None in sorting indices
                    sorting_indices_valid = ak.where(sorting_indices_filled == -1, None, sorting_indices_filled)
                    # Apply sorting indices to charges
                    inWindowPulseCharges_sorted = ak.take_along_axis(inWindowPulseCharges_filled, sorting_indices_valid,
                                                                     axis=1)
                    # Fill any None values resulting from indexing with -999
                    inWindowPulseCharges_sorted_filled = ak.fill_none(inWindowPulseCharges_sorted, -999)
                    # Convert to NumPy array
                    inWindowPulseCharges_np = ak.to_numpy(inWindowPulseCharges_sorted_filled).astype(np.float32)
                    inWindowPulseCharges_list.append(inWindowPulseCharges_np)
                idx_file += 1
            # Concatenate all arrays
            self._inWindowPulseCharges = np.concatenate(inWindowPulseCharges_list, axis=0)
        return self._inWindowPulseCharges

    def load_waveforms(self):
        """
        Load and return waveform data from all files.

        Returns:
            np.ndarray of np.float32: Concatenated array of waveforms, scaled to mV.
        """
        if self._waveforms is None:
            waveform_arrays = []
            for file_path in self.input_files:
                with uproot.open(file_path) as file:
                    waveforms_tree = file["waveforms"]
                    # Read waveform data directly as NumPy array
                    waveform_data = waveforms_tree["waveform"].array(library="np")
                    # Convert to float32 and scale
                    waveform_data = np.array(
                        [np.array(w, dtype=np.float32) for w in waveform_data]) * self.digitizer_bit_to_mV
                    waveform_arrays.append(waveform_data)
            # Concatenate along the first axis (n_events)
            self._waveforms = np.concatenate(waveform_arrays, axis=0)
        return self._waveforms

    def load_encoded_npe(self):
        """
        Load and preprocess the encoded_npe data.

        Returns:
            np.ndarray: The encoded_npe data.
        """
        nphotons = self.load_npe()
        encoded_npe = self.one_hot_encode_with_overflow(nphotons, self.npe_cut)
        return encoded_npe

    def load_good_data(self, nentries=100):
        """
        Load the waveforms, encoded_npe, and pe_times.
        Exclude waveforms where a true photon time is not finite, while printing a warning.

        Parameters:
            nentries (int): Desired fixed length for each pulse time array.

        Returns:
            np.ndarray, np.ndarray, np.ndarray: The waveforms, encoded_npe, and pe_times data.
        """
        waveforms = self.load_waveforms()
        encoded_npes = self.load_encoded_npe()
        pe_times = self.load_times(nentries)
        pmt_types = self.load_pmt_type()

        # Make a mask excluding infs and nans
        good_event_mask = np.all(np.isfinite(pe_times), axis=1)
        print(len(good_event_mask))

        # Check for bad pe times
        if np.sum(~good_event_mask) > 0:
            print('\n\n ---> WARNING!! Bad PE times present in the files!!! %d events will be removed.\n\n' % np.sum(
                ~good_event_mask))

        return waveforms[good_event_mask], encoded_npes[good_event_mask], pe_times[good_event_mask]
        
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
    def __init__(self, x, y, batch_size=2**13, shuffle=True, augment_data=True, max_shift_left=10, max_shift_right=10):
        """
        DataGenerator constructor.

        Parameters:
            x (np.ndarray): The input data for the model, 2d waveform data.
            y (np.ndarray): The true output data for the model, encoded_npe data or 2D time data.
            batch_size(int): The batch size.
            shuffle (bool): Whether to shuffle at the end of each epoch.
            augment_data (bool): Whether to shift waveform data to left and right, as augmentation.
            max_shift_left (int): Maximum number of bins to shift to the left when augmenting waveform data.
            max_shift_right (int): Maximum number of bins to shift to the right when augmenting waveform data.
        """
        self.x = x
        self.y = y
        assert max_shift_right>=0, "Maximum shift to the right should not be negative"
        self.max_shift_left = -1*abs(max_shift_left)
        self.max_shift_right = abs(max_shift_right)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_data = augment_data
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_x = self.x[indexes]
        batch_y = self.y[indexes]

        # Apply data augmentation
        if self.augment_data:
            batch_x = self._augment_data(batch_x)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _augment_data(self, batch_x):
        batch_size, seq_length = batch_x.shape
        # Generate random shifts
        shifts = np.random.randint(self.max_shift_left, self.max_shift_left + 1, batch_size)
        # Create an array with indices
        indices = np.mod(np.arange(seq_length) - shifts[:, None], seq_length)
        # Apply shifts
        augmented_batch_x = batch_x[np.arange(batch_size)[:, None], indices]
        return augmented_batch_x

