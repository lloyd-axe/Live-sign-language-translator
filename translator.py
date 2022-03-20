import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D

class ASLTranslator:
    def __init__(self):
        pass
    
class ModelingUtils:
    def __init__(self,
            input_shape,
            output_class_num,
            filters,
            kernels,
            strides):
        #Initial atributes
        self.input_shape = input_shape
        self.output_class_num = output_class_num
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self._num_conv_layers = len(filters)

        self.model = None
        self.build_model()
        self.compile_model()

    def build_model(self):
        self.model = Sequential()
        self._add_layers('conv')
        self._add_layers('flat')
        self._add_layers('dense')
        self.model.add(Dense(self.output_class_num, activation='softmax', name='output_layer'))

    def compile_model(self,
            optimizer = 'Adam',
            loss = 'categorical_crossentropy',
            metrics=['categorical_accuracy']):
        self.model.compile(optimizer=optimizer, 
                      loss=loss, 
                      metrics=metrics)

    def train(self, 
            model_input,
            val_data,
            batch_size = 10, 
            epochs = 30,
            early_stop = True,
            patience = 3,
            plot = True):
        callbacks = self._add_callbacks(early_stop, patience)
        history = self.model.fit(
                    model_input[0],
                    model_input[1], 
                    batch_size = batch_size,
                    epochs = epochs,
                    callbacks = callbacks,
                    validation_data = val_data)
        if plot:
            self._plot_train_history(history)
  
    def split_data(self, x_raw, y_raw, test_size = 0.25, val_size = 0.2, val_set = False):
        inputs = {}
        x = np.array(x_raw)
        y = to_categorical(np.array(y_raw))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
        
        if val_set:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = val_size)
            x_val = np.expand_dims(x_val, axis=3)
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        inputs['x_train'] = x_train
        inputs['x_test'] = x_test
        inputs['x_val'] = x_val
        inputs['y_train'] = y_train
        inputs['y_test'] = y_test
        inputs['y_val'] = y_val
        return inputs
    
    def save_model(self, model_name):
        self.model.save(model_name + '.h5')

    def _add_callbacks(self, early_stop, patience):
        if not os.path.isdir('train_checkpoints'):
            os.mkdir('train_checkpoints')
            print('train_checkpoints folder made')
        check_model_path = os.path.join('train_checkpoints', 'mdl-{epoch:02d}-{val_categorical_accuracy:2f}.hdf5')
        model_check = ModelCheckpoint(check_model_path, 
            monitor='val_categorical_accuracy', 
            verbose = 1, 
            save_best_only = True, 
            mode='max')
        callbacks = [model_check]
        if early_stop:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, verbose=3))
        return callbacks


    def _plot_train_history(self, history):
        plt.figure(figsize=(16,10))
        val = plt.plot(history.epoch, 
                history.history['val_categorical_accuracy'],
                '--',
                label='Val')
        plt.plot(history.epoch, 
                history.history['categorical_accuracy'], 
                color=val[0].get_color(), 
                label='Train')
        plt.plot(history.epoch, 
                history.history['loss'], 
                label='loss')
        plt.plot(history.epoch, 
                history.history['val_loss'], 
                label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('categorical_accuracy')
        plt.legend()
        plt.xlim([0,max(history.epoch)])
        
    def _add_layers(self, layer_type):
        if layer_type == 'conv':
            for layer_idx in range(self._num_conv_layers):
                self.model.add(Conv2D(
                    input_shape = self.input_shape,
                    filters = self.filters[layer_idx],
                    kernel_size = self.kernels[layer_idx],
                    activation = 'relu',
                    name = f'conv_layer_{layer_idx + 1}'
                ))
                self.model.add(MaxPool2D(
                    pool_size = self.kernels[layer_idx],
                    strides = self.strides[layer_idx],
                    padding = 'same'
                ))
                self.model.add(BatchNormalization(name = f'batchnormalization_{layer_idx + 1}'))
        elif layer_type == 'flat':
            self.model.add(Flatten(name='flat_layer'))
        elif layer_type == 'dense':
            self.model.add(Dense(self.filters[0]*2, activation='relu', name='dense_layer'))
        else:
            print(f'Cannot recognize {layer_type} layer.')

class ModelUtils:
    def __init__(self):
        pass