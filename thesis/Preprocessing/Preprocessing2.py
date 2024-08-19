#loading required packages
import pandas as pd
import numpy as np
import os
import re
import math
from PreprocessingFunctions import extract_timestamp_features,  get_open_cases

#indicate path to folder containing the original files
input_data_folder= "C:/Users/sarah/Documents/Information management/THESIS/labeled_logs_csv_processed"
#indicate path to folder where processed data will be saved
output_data_folder="C:/Users/sarah/Documents/Information management/THESIS/labeled_logs_csv_processed"
#specification name inut file
in_filename="Preprocessed1.csv"

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

data=data[static_cols+dynamic_cols]

#add features extracted from timestamps
data[timestamp_col] = pd.to_datetime(data[timestamp_col]) #conversion values in timestamps column to datetime objects
    #--> data is already in format that pandas can interpret as timestamps, but conversion can yield advantages during analysis
data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute #calculates time since midnight (expressed in minutes) for each timestamp by multiplying the hour component by 60 and adding the minute component
data["month"] = data[timestamp_col].dt.month #extracts month component each timestamp
data["weekday"] = data[timestamp_col].dt.weekday # extracts weekday component each timestamp --> integer value where 0= Monday.... 6= Sunday
data["hour"] = data[timestamp_col].dt.hour #extracts hour component each timestamp
data = data.groupby(case_id_col,group_keys=False).apply(extract_timestamp_features) #groups the dataframe by values of the case ID column and applies the funtion extract_timestamp_features to each group

#Calculate number open cases at a given date
def get_open_cases(date):
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))

#extracting inter case features
data = data.sort_values([timestamp_col], ascending=True, kind='mergesort') #sort the data by values in timestamp column in ascending order
dt_first_last_timestamps = data.groupby(case_id_col)[timestamp_col].agg([min, max]) #groups the data bu values in the case ID column and calculates the minimum and maximum timestamp for each group
dt_first_last_timestamps.columns = ["start_time", "end_time"] #result= dataframe with 2 columns: start_time and end_time
data["open_cases"] = data[timestamp_col].apply(get_open_cases) #function get_open_cases is applied to each value in the timestamps column and calculates the number of open cased at a given date based on first and last timestamp of each case
    #result = new column open cases
data.to_csv(os.path.join(output_data_folder, "Preprocessed2.csv"), sep=",", index=False)
