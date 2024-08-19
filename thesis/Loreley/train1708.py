#step 1: load packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from utils1808 import build_stateless_lstm, generator, get_n_steps
#definition build_stateless_lstm (from utils cluster)


#step 2:upload datasets

traces_train = np.load('C:/Users/sarah/Documents/Information management/THESIS/train1708.npy', allow_pickle=True)
traces_test = np.load('C:/Users/sarah/Documents/Information management/THESIS/test1708.npy', allow_pickle=True)
traces_validation = np.load('C:/Users/sarah/Documents/Information management/THESIS/validation1708.npy', allow_pickle=True)


min_inp_len = 1
batch_size = 32
# set this to None if non-binary target y
binary_y_index= True #mimicel data: hospitalized or not
# change the corresponding y shape: how many classes you predict
if binary_y_index:
    output_shape = 1 #mimicel 1: hospitalized or not
else:
    output_shape = 3 #BPIC2017: 4 potential outcomes: accepted/ cancelled/ denied/ pending --> predict 3 classes (variables: Concept:name_A_pending, Concept:name_A_Denied, Concept:name_A_Cancelled)

y_start_index = -output_shape #how many classes you predict (1: hospitalized or not)
#mimicel: y_start_index= -1 = the last column indicates the target class
#BPIC2017: y_start_index= -3 = the last 3 columns indicate the target classes

#Compute maximum length of traces across the 3 datasets
max_trace_len_all = max(max([len(trace) for trace in traces_train]),
                    max([len(trace) for trace in traces_test]), max([len(trace) for trace in traces_validation]))
#compute number of features in each trace of the training dataset (it is the same number for the test and validation set)
n_feature=traces_train[0].shape[1]

model = build_stateless_lstm(input_window_len=max_trace_len_all, n_feature=n_feature, output_shape=output_shape,
                             dropout=0.2, neurons_per_layer=[64, 64], masking=True)



#application generator function to each dataset
#Function used to feed the data in a batch manner
training_generator = generator(traces_train, y_start_index=y_start_index, max_trace_len_all=max_trace_len_all,
                            binary_y_index=binary_y_index, batch_size=batch_size, min_inp_len = min_inp_len, shuffle=True)
validation_generator = generator(traces_validation, y_start_index=y_start_index, max_trace_len_all=max_trace_len_all,
                               binary_y_index=binary_y_index, batch_size=batch_size, min_inp_len = min_inp_len, shuffle=True)
test_generator = generator(traces_test, y_start_index=y_start_index, max_trace_len_all=max_trace_len_all,
                         binary_y_index=binary_y_index, batch_size=batch_size, min_inp_len = min_inp_len, shuffle=True)


#application get_n_steps function to datasets
steps_per_epoch = get_n_steps(traces_train, batch_size=batch_size, min_inp_len=min_inp_len)
validation_steps = get_n_steps(traces_validation, batch_size=batch_size, min_inp_len=min_inp_len)
test_steps = get_n_steps(traces_test, batch_size=batch_size, min_inp_len=min_inp_len)

#initialization earlystopping callback object
    #Earlystopping= callback that stops training when a monitored metric has stopped improving
    #monitor: specifies the metric to monitor for improvements --> here val_loss: training will stop when the validation loss stops decreasing
    #patience: determines number of epochs with no improvements after which training will be stopped --> here 20: if the validation loss does not decrease for 20 consecutive epochs, the training will be stopped early
es = EarlyStopping(monitor='val_loss', patience=20)
filepath = 'C:/Users/sarah/Documents/Information management/THESIS/binary_all1708.keras'
#initialization modelcheckpoint callback object
    #Modelcheckpoint= callback that allows to save the model's weight during training based on certain conditions
    #filepath: where to save the model weights
    #monitor: specifies the metric to monitor for deciding whether to save the model weights --> here val_loss: validation loss is monitored
    #verbose=1: display a message when a model checkpoint is saved (while running obtain info)
    #save_best_only=True: only weights with the lowest validation loss are saved
mc = ModelCheckpoint(filepath=filepath, monitor="val_loss", verbose=1, save_best_only=True)


tb = TensorBoard(log_dir="./logs")

#log_dir = ".\logs\fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

#create callbacks lists that contains the earlystopping and modelcheckpoint objects
callbacks = [es, mc]

history = model.fit(x=training_generator, validation_data=validation_generator, epochs=100, callbacks=callbacks, verbose=1, shuffle=False, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
