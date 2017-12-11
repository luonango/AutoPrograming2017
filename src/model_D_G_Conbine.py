# encoding=utf8

import sys

reload(sys)
sys.setdefaultencoding('utf8')

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers.core import Masking, Dense, Activation, Dropout
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import model_from_json, Sequential
from keras.optimizers import RMSprop, Adam, SGD
import numpy as np
from keras.models import Sequential, Model


def build_generator(input_shape, output_length=20 * 100):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(units=128, activation='tanh', dropout=0.2, recurrent_dropout=0.))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Activation('sigmoid'))
    model.add(Dense(256))
    model.add(Dropout(0.15))
    model.add(Dense(output_length, kernel_initializer='normal'))
    model.add(Reshape(target_shape=(20, 100)))
    return model


def build_discriminator(input_shape=(20, 100)):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(units=128, activation='tanh', dropout=0.2, recurrent_dropout=0.))
    model.add(Dense(32, input_shape=input_shape))
    model.add(Activation('sigmoid'))
    model.add(Dense(16))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_combined(input_shape, generator, discriminator):
    input_layer = Input(shape=input_shape)
    glayer = generator(input_layer)
    discriminator.trainable = False
    final_layer = discriminator(glayer)
    combined = Model(input_layer, final_layer)
    return combined



# =========================================================================
# tools function for layer-net model
# =========================================================================


def compiler(layers_model):
    '''
    some compiler parameters
    '''
    _optimizer = RMSprop(lr=0.02, decay=1e-5)
    _loss = 'categorical_crossentropy'
    
    layers_model.compile(optimizer=_optimizer,
                         loss=_loss, metrics=['accuracy'])
    return layers_model


def trainer(model, train_x, train_y,
            batch_size=128,
            epochs=1,
            validation_split=0.0,
            auto_stop=False,
            best_record_path=None):
    # =========================================================================
    # set callbacks function for auto early stopping
    # by monitor the loss or val_loss if not change any more
    # =========================================================================
    callbacks = []
    
    if auto_stop == True:
        monitor = 'val_acc' if validation_split > 0.0 else 'acc'
        #         early_stopping = EarlyStopping(monitor=monitor, min_delta=0.001, patience=10, mode='auto')
        early_stopping = EarlyStopping(
            monitor=monitor, patience=20, mode='auto')
        callbacks.append(early_stopping)
    
    if best_record_path != None:
        monitor = 'val_acc' if validation_split > 0.0 else 'acc'
        check_pointer = ModelCheckpoint(
            best_record_path, monitor=monitor, verbose=1, save_best_only=True)
        callbacks.append(check_pointer)
    
    class MetricesHistory(Callback):
        def on_train_begin(self, logs={}):
            self.metrices = []
        
        def on_epoch_end(self, epoch, logs={}):
            if validation_split > 0.0:
                self.metrices.append((logs.get('loss'), logs.get(
                    'acc'), logs.get('val_loss'), logs.get('val_acc')))
            else:
                self.metrices.append((logs.get('loss'), logs.get('acc')))
    
    history = MetricesHistory()
    callbacks.append(history)
    model.fit(x=train_x, y=train_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              callbacks=callbacks)
    
    return model


def predictor(model, test_x, batch_size=64):
    # predict the test processed_data's labels with trained layer model
    output = model.predict(test_x, batch_size=batch_size)
    
    return output


def storageModel(model, frame_path, replace_record=True):
    record_path = None
    
    frameFile = open(frame_path, 'w')
    json_str = model.to_json()
    frameFile.write(json_str)  # save model's framework file
    frameFile.close()
    if replace_record == True:
        record_path = frame_path.replace('.json', '.h5')
        # save model's processed_data file
        model.save_weights(record_path, overwrite=True)
    
    return frame_path, record_path


def recompileModel(model):
    # optimizer = SGD(lr=0.1, decay=1e-5, nesterov=True)  # only CNNs_Net use
    # SGD
    optimizer = RMSprop(lr=0.02, decay=1e-5)
    
    # ps: if want use precision, recall and fmeasure, need to add these metrics
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[
        'accuracy', 'precision', 'recall', 'fmeasure'])
    return model


def loadStoredModel(frame_path, record_path, recompile=False):
    frameFile = open(frame_path, 'r')
    #     yaml_str = frameFile.readline()
    json_str = frameFile.readline()
    model = model_from_json(json_str)
    if recompile == True:
        model = recompileModel(model)  # if need to recompile
    model.load_weights(record_path)
    frameFile.close()
    
    return model
