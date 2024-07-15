import os
import tensorflow as tf
import keras
from keras import layers, models


class WaveformTransform(keras.layers.Layer):
    '''A custom layer to normalize the waveforms before each network evaluation so we don't have to do it manually
    before each time we train or evaluate the network'''

    def __init__(self, norm_mean=1800, norm_std=40):

        super().__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, waveforms):
        normalized_waveforms = (waveforms - self.norm_mean) / self.norm_std
        return normalized_waveforms


class PLSECounter():

    def __init__(self, waveform_shape, encoded_npe_shape, norm_mean=0, norm_std=40):
        self.waveform_length = waveform_shape[1]
        self.onehot_npe_length = encoded_npe_shape[1]
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.build_model()
        self.output_args = None

    def build_model(self):
        input_shape = (self.waveform_length, 1)
        input_layer = layers.Input(shape=input_shape)

        waveform_transform = WaveformTransform(norm_mean=self.norm_mean, norm_std=self.norm_std)
        normalized_waveforms = waveform_transform(input_layer)

        conv1 = layers.Conv1D(filters=8, kernel_size=4, activation='relu')(normalized_waveforms)
        conv2 = layers.Conv1D(filters=8, kernel_size=16, activation='relu')(normalized_waveforms)
        merged = layers.Concatenate(axis=-2)([conv1, conv2])

        flatten = layers.Flatten()(merged)
        dense1 = layers.Dense(128, activation='relu')(flatten)
        dense2 = layers.Dense(32, activation='relu')(dense1)
        dense3 = layers.Dense(32, activation='relu')(dense2)
        dense4 = layers.Dense(32, activation='relu')(dense3)

        output_layer = layers.Dense(self.onehot_npe_length, activation="softmax")(dense4)

        self.model = models.Model(input_layer, output_layer)
        self.model.summary()

    def compile_model(self, optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
                      metrics=["categorical_accuracy"]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def save(self, *args, **kwargs):
        '''Save the model as a .keras file'''
        assert self.output_args is not None, "Please run `verify_output_args` before attempting to save model."
        assert self.output_args['output_file'].endswith('.keras'), "Only keras format can be used for saving. For other types, use `export`."
        self.model.save(self.output_args['output_file'], *args, **kwargs)

    def export_tf(self, *args, **kwargs):
        '''Export the model as a TensorFlow saved model'''
        assert self.output_args is not None, "Please run `verify_output_args` before attempting to export model."
        tf_output_dir = os.path.join(self.output_args['output_dir'], 'tf_saved_model')
        self.model.export(tf_output_dir, format="tf_saved_model", *args, **kwargs)

    def verify_output_args(self, output_dir, filename="model.keras", overwrite=False):
        '''Verify that the output directory exists and that the output file won't be overwritten'''
        # Convert relative/symbolic path to an absolute path
        output_dir = os.path.realpath(output_dir)

        # If the output directory doesn't exist, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Construct the full output file path
        output_file = os.path.join(output_dir, filename)

        # Check if the file already exists
        if os.path.exists(output_file) and not overwrite:
            assert False, "Output file (%s) already exists. Use 'overwrite=True' to overwrite it."%output_file
        else:
            # File is able to be saved, save location information for future use
            self.output_args = {
                'output_dir': output_dir,
                'output_file': output_file,
            }
            return True
