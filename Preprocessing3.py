#loading required packages
import pandas as pd
import numpy as np
import os
import re
import math
from PreprocessingFunctions import label_cases

#indicate path to folder containing the original files
input_data_folder= "C:/Users/sarah/Documents/Information management/THESIS/labeled_logs_csv_processed"
#indicate path to folder where processed data will be saved
output_data_folder="C:/Users/sarah/Documents/Information management/THESIS/labeled_logs_csv_processed"
#specification name inut file
in_filename="Preprocessed2.csv"

#define column names and labels for processing the dataset
case_id_col= "stay_id"
activity_col= "activity"
timestamp_col= "timestamps"
label_col= "label"
pos_label= "deviant"
neg_label= "regular"

#set category frequency threshold = only the categories with frequency higher than X are considered
category_freq_threshold = 10 #can still change this number based on how many categories are eliminated by this threshold

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

#impute missing values by using forward filling method
grouped = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col) #sort the data by timestamp colum = ensure the data within each group (= stay_id) is ordered chronologically
for col in static_cols + dynamic_cols:
    data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
    #for each group, missing values in the static_cols and dynamic_cols are filled using forward fill (= replace missing value wiht the last known value in the column within each group)
    data[cat_cols] = data[cat_cols].fillna('missing') #for categorical columns, missing values are filled with the string 'missing'
    data = data.fillna(0) #any remaining missing values are filled with 0

#set infrequent (= less than 10) factor levels to other
for col in cat_cols: #looping over all categorical columns
    counts = data[col].value_counts() #calculate value counts for each category
    mask = data[col].isin(counts[counts >= category_freq_threshold].index) #create a mask for infrequent categories
        #Identify rows where the category occurs less frequent than threshold (category_freq_threshold)
        #select categories wiht counts greater than or equal to threshold and creates boolean maks based on whether each value in the column is in that selected set of categories
    data.loc[~mask, col] = "other" #Replace values  in the column 'col' with other for the rows where the mask is False


#apply labeling function: case should be labeled as deviant if it contains a hadm_id value
dt_labeled=data.sort_values(timestamp_col, ascending=True, kind="mergesort").groupby(case_id_col, group_keys=False).apply(label_cases)
dt_labeled.to_csv(os.path.join(output_data_folder, "Preprocessed_mimic.csv"), sep=",", index=False)
