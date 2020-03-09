# path
import os
from os.path import isdir, join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, callbacks
from tensorflow.keras.utils import plot_model
from guppy import hpy
import random
import copy
import librosa
import time
import math
import sys
import scipy.fftpack
from scipy import signal
from scipy.io import wavfile
from sklearn.preprocessing import normalize
h = hpy()


SAMPLE_LIMIT = 2000

TRAIN_AUDIO_PATH = 'train/train/audio/'
TARGET_LIST = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

OUTPUTS = len(TARGET_LIST) + 2

INPUT_SAMPLES = 8192
NOISE_MULTIPLIER = 1 # How many noised copies should we make
UNKNOWN_COUNT = SAMPLE_LIMIT * (NOISE_MULTIPLIER + 1)
SILENCE_COUNT = SAMPLE_LIMIT * (NOISE_MULTIPLIER + 1)
NOISE_RATIO = .2

VALID_SPLIT = .2

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 100
DROPOUT_RATE = 0.5

PATIENCE = 40

TYPE_REGULAR = 0
TYPE_NOISED = 1
TYPE_UNKNOWN = 2
TYPE_SILENCE = 3

class Transform:
    def __init__(self, name, filename, size):
        self.name = name
        self.filename = filename
        self.size = size

    def calc(self, array):
        pass

class FFT(Transform):
    def __init__(self):
        Transform.__init__(self, "fft", "xfft.npy", INPUT_SAMPLES // 2 + 1)

    def calc(self, array):
        return np.abs(np.fft.rfft(array))

class DHT(Transform):
    def __init__(self):
        Transform.__init__(self, "dht", "xdht.npy", INPUT_SAMPLES)

    DHT_MTX = None
    def calc(self, array):
        if DHT.DHT_MTX is None:
            N = len(array)
            ts = np.arange(N) / N
            fs = np.arange(N)
            args = np.outer(ts, fs)
            DHT.DHT_MTX = np.cos(2 * math.pi * args) + np.sin(2 * math.pi * args)
        return np.dot(DHT.DHT_MTX, array)

class DCT(Transform):
    def __init__(self):
        Transform.__init__(self, "dct", "xdct.npy", INPUT_SAMPLES)

    def calc(self, array):
        return scipy.fftpack.dct(array)


transforms = [FFT(), DHT(), DCT()]



def label_to_id(label):
    if label in TARGET_LIST:
        return TARGET_LIST.index(label)
    elif label == "unknown":
        return len(TARGET_LIST)
    elif label == "silence":
        return len(TARGET_LIST) + 1

def shuffle(*arrays):
    state = np.random.get_state()
    for array in arrays:
        np.random.set_state(state)
        np.random.shuffle(array)

def generate_convolution(size):
    input = Input(shape=(size, 1))
    x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides='1')(input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides='1')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides='1')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    return input, x

background_noise = []
def load_metadata_from_wavs():
    global background_noise
    background = [f for f in os.listdir(join(TRAIN_AUDIO_PATH, '_background_noise_')) if f.endswith('.wav')]
    for wav in background:
        samples, sample_rate = librosa.load(join(join(TRAIN_AUDIO_PATH, '_background_noise_'), wav), sr=INPUT_SAMPLES)
        background_noise.append(samples)

    dirs = [f for f in os.listdir(TRAIN_AUDIO_PATH) if isdir(join(TRAIN_AUDIO_PATH, f))]
    dirs.sort()

    wavs = []
    labels = []
    unknown_wavs = []

    unknown_list = [d for d in dirs if d not in TARGET_LIST and d != '_background_noise_']
    print('target_list : ', end='')
    print(TARGET_LIST)
    print('unknowns_list : ', end='')
    print(unknown_list)
    print('silence : _background_noise_')

    i = 0
    for directory in dirs[1:]:
        waves = [f for f in os.listdir(join(TRAIN_AUDIO_PATH, directory)) if f.endswith('.wav')]

        for j, wav in enumerate(waves):
            # samples, sample_rate = librosa.load(join(join(TRAIN_AUDIO_PATH, directory), wav), sr=16000)
            sample_rate, samples = wavfile.read(join(join(TRAIN_AUDIO_PATH, directory), wav))
            samples = np.concatenate((np.zeros((INPUT_SAMPLES - 8000) // 2, dtype="float32"),
                                      samples[::2],
                                      np.zeros((INPUT_SAMPLES - 8000) // 2, dtype="float32")))
            if len(samples) != INPUT_SAMPLES:
                continue

            if directory in unknown_list:
                unknown_wavs.append((wav, directory))
            else:
                wavs.append((TYPE_REGULAR, wav))
                labels.append(directory)

    wavc = len(wavs)
    for n in range(NOISE_MULTIPLIER):
        for i in range(wavc):
            wavs.append((TYPE_NOISED, wavs[i][1]))
            labels.append(labels[i])

    for i in range(UNKNOWN_COUNT):
        wavs.append((TYPE_UNKNOWN, random.choice(unknown_wavs)))
        labels.append("unknown")

    for i in range(SILENCE_COUNT):
        wavs.append((TYPE_SILENCE, random.randrange(0, len(background_noise))))
        labels.append("silence")

    return wavs, labels

def get_one_noise():
    noise_num = random.randrange(0, len(background_noise))
    selected_noise = background_noise[noise_num]
    start_idx = random.randint(0, len(selected_noise) - 1 - INPUT_SAMPLES)
    return selected_noise[start_idx:(start_idx + INPUT_SAMPLES)]

wavs, labels = load_metadata_from_wavs()
NUM_SAMPLES = 0

def generator():
    global wavs, labels, NUM_SAMPLES
    NUM_SAMPLES = len(wavs)
    while True:
        sh = list(zip(wavs, labels))
        random.shuffle(sh)
        swavs, slabels = zip(*sh)
        swavs = list(swavs)
        slabels = list(slabels)
        for offset in range(0, NUM_SAMPLES, BATCH_SIZE):
            # Get the samples you'll use in this batch
            wav_samples = swavs[offset:offset + BATCH_SIZE]
            label_samples = slabels[offset:offset + BATCH_SIZE]
            for b in range(len(wav_samples)):
                directory = label_samples[b]
                label_samples[b] = label_to_id(directory)
                wav_data = wav_samples[b]
                type_ = wav_data[0]
                if type_ == TYPE_REGULAR or type_ == TYPE_NOISED:
                    sample_rate, samples = wavfile.read(join(join(TRAIN_AUDIO_PATH, directory), wav_data[1]))
                    samples = np.concatenate((np.zeros((INPUT_SAMPLES - 8000) // 2, dtype="float32"),
                                              samples[::2],
                                              np.zeros((INPUT_SAMPLES - 8000) // 2, dtype="float32")))
                    if type_ == TYPE_NOISED:
                        samples += get_one_noise() * NOISE_RATIO
                    frequencies, times, spectrogram = signal.spectrogram(samples, 8000)
                    wav_samples[b] = spectrogram
                elif type_ == TYPE_UNKNOWN:
                    sample_rate, samples = wavfile.read(join(join(TRAIN_AUDIO_PATH, wav_data[1][1]), wav_data[1][0]))
                    samples = np.concatenate((np.zeros((INPUT_SAMPLES - 8000) // 2, dtype="float32"),
                                              samples[::2],
                                              np.zeros((INPUT_SAMPLES - 8000) // 2, dtype="float32")))
                    frequencies, times, spectrogram = signal.spectrogram(samples, 8000)
                    wav_samples[b] = spectrogram
                elif type_ == TYPE_SILENCE:
                    frequencies, times, spectrogram = signal.spectrogram(get_one_noise() * .5, 8000)
                    wav_samples[b] = spectrogram
                wav_samples[b] = wav_samples[b].reshape((129, 36, 1))
                label_samples[b] = keras.utils.to_categorical(label_samples[b], len(TARGET_LIST) + 2)
            wav_samples = np.array(wav_samples)
            label_samples = np.array(label_samples)

            yield wav_samples, label_samples





if __name__ == "__main__":
    applied_transforms = []
    for i in range(1, len(sys.argv)):
        applied_transforms.extend(tr for tr in transforms if tr.name==sys.argv[i])

    print("Used transforms:", applied_transforms)

    # Callbacks

    early_stopping = callbacks.EarlyStopping(patience=PATIENCE, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(filepath="weights.hdf5", save_best_only=True, verbose=1)



    input_x = Input(shape=(129, 36, 1))
    x = layers.BatchNormalization()(input_x)
    x = layers.Conv2D(32, (7, 3), padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (7, 3), padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output_x = layers.Dense(OUTPUTS, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_x, outputs=[output_x,])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['accuracy'])




    model.summary()

    train_generator = generator()
    valid_generator = generator()

    a = next(train_generator)

    plot_model(model, show_shapes=True, show_layer_names=False)

    #X = X.reshape((-1, 129, 36, 1))
    history = model.fit_generator(train_generator,
                                  validation_data=valid_generator,
                                  validation_steps=2,
                                  epochs=EPOCHS,
                                  steps_per_epoch=NUM_SAMPLES // BATCH_SIZE,
                                  callbacks=[early_stopping, checkpoint],
                                  verbose=1)

    model.load_weights("weights.hdf5")
    #print(model.evaluate(X, Y))

    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()"""
