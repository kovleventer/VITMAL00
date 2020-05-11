from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, SeparableConv1D, ReLU, Add, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    x = input_data
    for r in range(recur_layers):
        x = GRU(units, return_sequences=True, 
                 implementation=2)(x)
        x = BatchNormalization()(x)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(x)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, ))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, output_dim=29):
    """ Build a deep network for speech
    """
    lstm_sizes = [32, 32]
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    x = conv_1d
    for size in lstm_sizes:
        x = Bidirectional(LSTM(size, return_sequences=True, ))(x)
        x = BatchNormalization()(x)
    time_dense = TimeDistributed(Dense(output_dim))(x)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def model_deep_bidir_sepconv(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, output_dim=29, recurrent_sizes=[32, 32]):
    """ Build a deep network for speech
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = SeparableConv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    x = conv_1d
    for size in recurrent_sizes:
        x = Bidirectional(GRU(size, activation='relu',
        return_sequences=True, implementation=2, ))(x)
        x = BatchNormalization()(x)
    time_dense = TimeDistributed(Dense(output_dim))(x)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def model_deep_bidir_deep_sepconv(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, output_dim=29, recurrent_sizes = [32, 32]):
    """ Build a deep network for speech
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    
    x = input_data
    for filter_ in filters:
        x = SeparableConv1D(filter_, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu')(x)
        
        
    for size in recurrent_sizes:
        x = Bidirectional(GRU(size, activation='relu',
        return_sequences=True, implementation=2, ))(x)
        x = BatchNormalization()(x)
    time_dense = TimeDistributed(Dense(output_dim))(x)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(cnn_output_length(x, kernel_size, conv_border_mode, conv_stride), kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def jasper_model(input_dim, output_dim=29, R=2, B=3):
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_prolog = Conv1D(256, 11, strides=1, padding="same")(input_data)
    conv_prolog = BatchNormalization()(conv_prolog)
    conv_prolog = ReLU()(conv_prolog)

    x = conv_prolog
    for b in range(B):
        before_r = x
        for r in range(R):
            x = Conv1D(256 + 128 * b, 9 + 2 * b if b != 0 else 11, padding="same")(x)
            x = BatchNormalization()(x)
            if r == R-1:
                before_r = Conv1D(256 + 128*b, 1, padding="same")(before_r)
                before_r = BatchNormalization()(before_r)
                x = Add()([x, before_r])
            x = ReLU()(x)
            x = Dropout(0.3)(x)
    conv_epilog = Conv1D(896, 19, padding="same", dilation_rate=2)(x)
    conv_epilog = BatchNormalization()(conv_epilog)
    conv_epilog = ReLU()(conv_epilog)

    conv_next = Conv1D(1024, 1, padding="same")(conv_epilog)
    conv_next = BatchNormalization()(conv_next)
    conv_next = ReLU()(conv_next)

    conv_final = Conv1D(output_dim, 1, padding="same")(conv_next)
    #conv_final = Concatenate()([conv_final, conv_final])
    
    y_pred = Activation('softmax', name='softmax')(conv_final)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def quartznet_model(input_dim, output_dim=29, R=2, B=3):
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_prolog = SeparableConv1D(256, 33, strides=2, padding="same")(input_data)
    conv_prolog = BatchNormalization()(conv_prolog)
    conv_prolog = ReLU()(conv_prolog)

    x = conv_prolog
    for b in range(B):
        before_r = x
        for r in range(R):
            x = SeparableConv1D(256, 33, padding="same")(x)
            x = BatchNormalization()(x)
            if r == R - 1:
                before_r = SeparableConv1D(256, 1, padding="same")(before_r)
                before_r = BatchNormalization()(before_r)
                x = Add()([x, before_r])
            x = ReLU()(x)

    conv_epilog = SeparableConv1D(512, 51, padding="same")(x)
    conv_epilog = BatchNormalization()(conv_epilog)
    conv_epilog = ReLU()(conv_epilog)

    conv_next = SeparableConv1D(1024, 1, padding="same")(conv_epilog)
    conv_next = BatchNormalization()(conv_next)
    conv_next = ReLU()(conv_next)

    conv_final = SeparableConv1D(output_dim, 1, padding="same", dilation_rate=2)(conv_next)

    y_pred = Activation('softmax', name='softmax')(conv_final)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x/2
    print(model.summary())
    return model

def sep1d_bn_relu(x, filters, kernel_size, strides=1, dilation_rate=1, padding="same"):
    x = SeparableConv1D(filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def quartznet_12x1(input_dim, output_dim=29):
    # No bn and relu activations, so probably not a good model
    input_data = Input(name='the_input', shape=(None, input_dim))
    x = input_data
    x = sep1d_bn_relu(x, 256, 33, strides=2, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 256, 39, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 39, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 39, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 512, 51, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 512, 51, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 512, 51, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 512, 63, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 512, 63, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 512, 63, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 512, 75, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 1024, 1, strides=1, dilation_rate=1, padding="same")

    conv_final = SeparableConv1D(output_dim, 1, padding="same", dilation_rate=2)(x)
    y_pred = Activation('softmax', name='softmax')(conv_final)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x / 2
    print(model.summary())
    return model

def quartznet_12x1_15_39(input_dim, output_dim=29):
    # No bn and relu activations, so probably not a good model
    input_data = Input(name='the_input', shape=(None, input_dim))
    x = input_data
    x = sep1d_bn_relu(x, 128, 15, strides=2, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 128, 15, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 128, 15, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 128, 15, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 128, 15, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 128, 21, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 128, 21, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 128, 21, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 256, 27, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 27, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 27, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")

    x = sep1d_bn_relu(x, 256, 39, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 512, 1, strides=1, dilation_rate=1, padding="same")

    conv_final = SeparableConv1D(output_dim, 1, padding="same", dilation_rate=2)(x)
    y_pred = Activation('softmax', name='softmax')(conv_final)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x / 2
    print(model.summary())
    return model



def quartznet_15x5(input_dim, output_dim=29):
    input_data = Input(name='the_input', shape=(None, input_dim))
    x = input_data
    x = sep1d_bn_relu(x, 256, 33, strides=2, dilation_rate=1, padding="same")
    for _ in range(15):
        x = sep1d_bn_relu(x, 256, 33, strides=1, dilation_rate=1, padding="same")
    for _ in range(15):
        x = sep1d_bn_relu(x, 256, 39, strides=1, dilation_rate=1, padding="same")
    for _ in range(15):
        x = sep1d_bn_relu(x, 512, 51, strides=1, dilation_rate=1, padding="same")
    for _ in range(15):
        x = sep1d_bn_relu(x, 512, 63, strides=1, dilation_rate=1, padding="same")
    for _ in range(15):
        x = sep1d_bn_relu(x, 512, 75, strides=1, dilation_rate=1, padding="same")
    x = sep1d_bn_relu(x, 512, 87, strides=1, dilation_rate=2, padding="same")
    x = sep1d_bn_relu(x, 1024, 1, strides=1, dilation_rate=1, padding="same")
    conv_final = SeparableConv1D(output_dim, 1, padding="same", dilation_rate=2)(x)
    y_pred = Activation('softmax', name='softmax')(conv_final)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x / 2
    print(model.summary())
    return model


