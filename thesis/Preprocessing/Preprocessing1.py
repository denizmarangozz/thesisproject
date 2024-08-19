#loading required packages
import pandas as pd
import numpy as np
import os
import re
import math
from PreprocessingFunctions import transform_rangevalues, transform_scalevalues, round_down, retain_numeric_values, lowercase

#indicate path to folder containing the original files
input_data_folder= "C:/Users/sarah/Documents/Information management/THESIS/orig_logs"
#indicate path to folder where processed data will be saved
output_data_folder="C:/Users/sarah/Documents/Information management/THESIS/labeled_logs_csv_processed"
#specification name inut file
in_filename="mimicel.csv"

#define column names and labels for processing the dataset
case_id_col= "stay_id"
activity_col= "activity"
timestamp_col= "timestamps"
label_col= "label"
pos_label= "deviant"
neg_label= "regular"


#define features for classifier
dynamic_cat_cols = ['activity', 'arrival_transport', 'disposition',
                   'icd_title', 'rhythm', 'name'] #categorical event attributes
static_cat_cols = ['gender', 'race', 'chiefcomplaint'] #categorical case attributes (= describe individual cases and are the same across all events associated with particular case)
dynamic_num_cols = ['temperature', 'heartrate',
                    'resprate', 'o2sat', 'sbp', 'dbp', 'pain','gsn', 'ndc', 'etccode'] #numerical event attributes
static_num_cols = ['subject_id','hadm_id', 'acuity'] #numerical case attributes

#concatening lists
static_cols = static_cat_cols + static_num_cols + [case_id_col] #all case columns combined
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col] #all event columns combined
cat_cols = dynamic_cat_cols + static_cat_cols #all categorical variables combined

#read csv file specified by in-filename from folder specified in input_dat_folder and store the dataframe as data using ; as seperator
data = pd.read_csv(os.path.join(input_data_folder, in_filename), sep=",")

#delete irrelevant values in pain column
data['pain']= data['pain'].apply(transform_rangevalues)
data['pain']=data['pain'].apply(transform_scalevalues)
data['pain']=data['pain'].apply(round_down)
data['pain']= retain_numeric_values(data['pain'])

#convert values in column name and chiefcomplaint to lowercase
lowercase(data, ['name', 'chiefcomplaint'])

#exporting the data (saving)
data.to_csv(os.path.join(output_data_folder, "Preprocessed1.csv"), sep=",", index=False)
