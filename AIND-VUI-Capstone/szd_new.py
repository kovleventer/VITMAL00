# allocate 50% of GPU memory (if you like, feel free to change this)
import tensorflow as tf
from keras.callbacks import LearningRateScheduler

print(tf.__version__)
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD
import math


config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config))

# watch for any changes in the sample_models module, and reload it automatically
# import NN architectures for speech recognition
from sample_models import *
# import function for training acoustic model
from train_utils import train_model

from wer import wer
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
import numpy as np


def calculate_wer2(input_to_softmax, model_path, words=False):
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()
    wers = []
    for index in range(len(data_gen.valid_texts)):
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))

        input_to_softmax.load_weights(model_path)
        prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
        output_length = [input_to_softmax.output_length(data_point.shape[0])]
        pred_ints = (K.eval(K.ctc_decode(
            prediction, output_length)[0][0]) + 1).flatten().tolist()

        pred = ''.join(int_sequence_to_text(pred_ints))

        if words:
            w = wer(transcr.split(), pred.split())
        else:
            w = wer(list(transcr), list(pred))
        wers.append(w)
        if index % 100 == 0:
            print(index, len(data_gen.valid_texts), wers[-1])

    print("FINAL WER:", sum(wers) / len(wers), "words:", words)

initial_lrate = 0.02
epochs = 20

def exp_decay(t):
   k = 0.1
   lrate = initial_lrate * math.exp(-k*t)
   #print("LRATE:", lrate)
   return lrate

def step_decay(epoch):
   drop = 0.5
   epochs_drop = 5.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

for i in range(epochs):
    print(i, exp_decay(i))

for i in range(epochs):
    print(i, step_decay(i))

#exit()

def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator()
    data_gen.load_train_data()
    data_gen.load_validation_data()

    # obtain the true transcription and the audio features
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
        prediction, output_length)[0][0]) + 1).flatten().tolist()

    print('True transcription:\n' + '\n' + transcr)
    print('-' * 80)
    print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('-' * 80)


get_predictions(index=0,
                partition='validation',
                input_to_softmax=quartznet_12x1_15_39(161),
                model_path='results/quartz_1212151.h5')
#exit()

quartz_1212151 = quartznet_12x1_15_39(161)
train_model(input_to_softmax=quartz_1212151,
            pickle_path='quartz_1212151.pickle',
            save_model_path='quartz_1212151.h5',
            spectrogram=True, min_duration=10.0, max_duration=10000., minibatch_size=5, wer=None, epochs=epochs,
           # optimizer=Adam(lr=initial_lrate),
            #optimizer=SGD(lr=initial_lrate, momentum=0.9, nesterov=True, clipnorm=5),
            #lratedecay=LearningRateScheduler(step_decay)
)


