import os
import pickle
import pandas as pd
import numpy as np
import time
from DatasetManager1708 import DatasetManager1708

# Function to create directories if they do not exist
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


dataset_name="mimicel"
results_dir="C:/Users/sarah/Documents/Information management/THESIS"
# Check and create directories if they do not exist
create_directory_if_not_exists(results_dir)

#set fixed parameters
train_ratio= 0.8
validation_ratio=0.1
random_state= 22

dataset_manager=DatasetManager1708(dataset_name)
data=dataset_manager.read_dataset()
# Define cls_encoder_args
cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': True}

print(data.dtypes)


#transform categorical variables using cat_encoding
data=dataset_manager.cat_encoding(data)
dataOrder=dataset_manager.reorder_columns(data)
train,test,validation= dataset_manager.split_data_strict(dataOrder,train_ratio,validation_ratio, split='temporal')
testcase=dataset_manager.split_event_log_by_case(test)
testarray=dataset_manager.dict_to_array(testcase)
#debugging statements
first_cell_value_test = testarray.flat[0]
second_cell_value_test=testarray.flat[1]
n_feature_test=testarray[0].shape[1]

traincase=dataset_manager.split_event_log_by_case(train)
trainarray=dataset_manager.dict_to_array(traincase)
#debugging statements
first_cell_value_train = trainarray.flat[0]
second_cell_value_train=trainarray.flat[1]
n_feature_train=trainarray[0].shape[1]

validationcase=dataset_manager.split_event_log_by_case(validation)
validationarray=dataset_manager.dict_to_array(validationcase)
#debugging statements
first_cell_value_val = validationarray.flat[0]
second_cell_value_val=validationarray.flat[1]
n_feature_val=validationarray[0].shape[1]

np.save("C:/Users/sarah/Documents/Information management/THESIS/validation1708.npy", validationarray)
np.save("C:/Users/sarah/Documents/Information management/THESIS/test1708.npy", testarray)
np.save("C:/Users/sarah/Documents/Information management/THESIS/train1708.npy", trainarray)

