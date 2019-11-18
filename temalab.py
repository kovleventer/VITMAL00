# path
import os
from os.path import isdir, join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers
from guppy import hpy
import random
import copy
import librosa
import time
import math
import sys
h = hpy()



TRAIN_AUDIO_PATH = 'train/train/audio/'
TARGET_LIST = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

OUTPUTS = len(TARGET_LIST) + 2

INPUT_SAMPLES = 8192
NOISE_MULTIPLIER = 1 # How many noised copies should we make
UNKNOWN_COUNT = 2000 * (NOISE_MULTIPLIER + 1)
SILENCE_COUNT = 2000 * (NOISE_MULTIPLIER + 1)
NOISE_RATIO = .2

VALID_SPLIT = .2

LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 100
DROPOUT_RATE = 0.5

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


transforms = [FFT(), DHT()]



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

def load_data_from_wavs(transforms):

    background = [f for f in os.listdir(join(TRAIN_AUDIO_PATH, '_background_noise_')) if f.endswith('.wav')]
    background_noise = []
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
        start = time.time()
        waves = [f for f in os.listdir(join(TRAIN_AUDIO_PATH, directory)) if f.endswith('.wav')]
        i = i + 1
        print(str(i) + ":" + str(directory) + " ", end="", flush=True)
        for wav in waves:
            samples, sample_rate = librosa.load(join(join(TRAIN_AUDIO_PATH, directory), wav), sr=16000)
            samples = np.concatenate((np.zeros((INPUT_SAMPLES-8000)//2, dtype="float32"),
                                     samples[::2],
                                     np.zeros((INPUT_SAMPLES-8000)//2, dtype="float32")))
            # samples1 = librosa.resample(samples, sample_rate, 8000)
            # retek = samples2 - samples1
            if len(samples) != INPUT_SAMPLES:
                continue



            if directory in unknown_list:
                unknown_wavs.append(samples)
            else:
                wavs.append(samples)
                labels.append(directory)
        end = time.time()
        print(end - start, flush=True)

    cnt = len(wavs) * (NOISE_MULTIPLIER + 1) + UNKNOWN_COUNT + SILENCE_COUNT
    X = np.zeros((cnt, INPUT_SAMPLES), dtype="float32")
    Y = np.zeros((cnt, 1), dtype="int32")

    def get_one_noise():
        noise_num = random.randrange(0, len(background_noise))
        selected_noise = background_noise[noise_num]
        start_idx = random.randint(0, len(selected_noise) - 1 - INPUT_SAMPLES)
        return selected_noise[start_idx:(start_idx + INPUT_SAMPLES)]


    for i in range(len(wavs)):
        X[i] = wavs[i]
        Y[i] = label_to_id(labels[i])
    acc = len(wavs)

    for n in range(NOISE_MULTIPLIER):
        for i in range(len(wavs)):
            X[acc + n * len(wavs) + i] = get_one_noise() * NOISE_RATIO + wavs[i]
            Y[acc + n * len(wavs) + i] = label_to_id(labels[i])
    acc += len(wavs) * NOISE_MULTIPLIER

    for i in range(UNKNOWN_COUNT):
        X[acc + i] = random.choice(unknown_wavs)
        Y[acc + i] = label_to_id("unknown")
    acc += UNKNOWN_COUNT

    for i in range(SILENCE_COUNT):
        X[acc + i] = get_one_noise() * .5
        Y[acc + i] = label_to_id("silence")

    Xs = []
    for transform in transforms:
        Xs.append(np.zeros((cnt, transform.size), dtype="float32"))
        for i in range(cnt):
            Xs[-1][i] = transform.calc(X[i])



    print(h.heap())
    return X, Y, Xs


if __name__ == "__main__":
    applied_transforms = []
    for i in range(1, len(sys.argv)):
        applied_transforms.extend(tr for tr in transforms if tr.name==sys.argv[i])

    print("Used transforms:", applied_transforms)

    xfile = "x.npy"
    yfile = "y.npy"
    try:
        X = np.load(xfile)
        Y = np.load(yfile)
        Xs = []
        for transform in applied_transforms:
            Xs.append(np.load(transform.filename))
        print("Loading arrays from file")
    except:
        X, Y, Xs = load_data_from_wavs(applied_transforms)
        np.save(xfile, X)
        np.save(yfile, Y)
        for i, transform in enumerate(applied_transforms):
            np.save(transform.filename, Xs[i])

    shuffle(X, Y, *Xs)

    # For Conv1D add Channel
    X = X.reshape((-1, INPUT_SAMPLES, 1))
    for i, transform in enumerate(applied_transforms):
        Xs[i] = Xs[i].reshape((-1, transform.size, 1))

    # Make Label data 'class num' -> 'One hot vector'
    Y = keras.utils.to_categorical(Y, len(TARGET_LIST) + 2)


    # Conv1D Model
    input_x = Input(shape=(INPUT_SAMPLES, 1))
    x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)

    input_tensors = [input_x,]
    transform_tensors = [x,]

    for transform in applied_transforms:
        input_xtr, x_tr = generate_convolution(transform.size)
        input_tensors.append(input_xtr)
        transform_tensors.append(x_tr)

    if len(applied_transforms) > 0:
        x = keras.layers.Concatenate()(transform_tensors)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output_tensor = layers.Dense(OUTPUTS, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensors, outputs=[output_tensor,])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['accuracy'])


    model.summary()


    history = model.fit([X, *Xs], Y, validation_split=VALID_SPLIT, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, verbose=1)


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
    plt.show()
