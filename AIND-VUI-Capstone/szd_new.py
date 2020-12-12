seed_value= 0
#1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
#tf.random.set_seed(seed_value)
# for later versions:
tf.random.set_random_seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# allocate 50% of GPU memory (if you like, feel free to change this)
import tensorflow as tf
from keras.callbacks import LearningRateScheduler

print(tf.__version__)
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD
import math
from adamw import AdamW
from wer_k import calculate_wer2


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

from keras.optimizers import Nadam, Adagrad


from wer_k import WERCallback
from cosine_annealing import *

initial_lrate = 0.02
epochs = 10

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

MODEL_NAME = "quartznet_12x1_15_39"

#calculate_wer2(input_to_softmax=quartznet_12x1_15_39(161), model_path='results/quartznet_12x1_15_39.h5', words=True)
#calculate_wer2(input_to_softmax=quartznet_12x1_15_39(161), model_path='results/quartznet_12x1_15_39.h5', words=False)
#exit()

#get_predictions(1, "validation", quartznet_12x1_15_39(161), "results/quartznet_12x1_15_39.h5")
#get_predictions(2, "validation", quartznet_12x1_15_39(161), "results/quartznet_12x1_15_39.h5")
#get_predictions(3, "validation", quartznet_12x1_15_39(161), "results/quartznet_12x1_15_39.h5")
#get_predictions(4, "validation", quartznet_12x1_15_39(161), "results/quartznet_12x1_15_39.h5")

#exit()

for lrate in [
    #0.025, 0.022,
    #0.018, 0.014,
    # 0.010
    0.02]:
    quartz_1212151 = quartznet_12x1_15_39(161) # 161
    #exit()
    #quartz_1212151.load_weights("results/quartznet_12x1_15_39.h5")

    #exit()
    train_model(input_to_softmax=quartz_1212151,
                pickle_path=MODEL_NAME + ".pickle",
                save_model_path=MODEL_NAME + ".h5",
                train_json='train_100_corpus.json',
                spectrogram=True,
                min_duration=10.0, max_duration=10000., minibatch_size=5, cbs=[
            WERCallback(quartz_1212151, 'results/' + MODEL_NAME + '.h5'),
            #LinearAnnealingScheduler(T_max=4, eta_max=initial_lrate, eta_min=1e-4)
            ExponentionalAnnealingScheduler(T_max=4, eta_max=initial_lrate)
        ],
                epochs=epochs,
                #optimizer=Adagrad(lr=lrate),
                #optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                #lratedecay=LearningRateScheduler(step_decay)
    )


