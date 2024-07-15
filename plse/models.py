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
        self.model.save(*args, **kwargs)
