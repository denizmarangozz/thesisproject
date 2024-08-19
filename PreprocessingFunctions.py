# loading required packages
import pandas as pd
import numpy as np
import os
import re
import math

timestamp_col= "timestamps"
label_col= "label"
pos_label= "deviant"
neg_label= "regular"

#extract timestamp related features from a group of events
def extract_timestamp_features(group):
    #sort the group by timestamp in descending order
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')

    #calculate time since last event --> result= new column named 'timesincelastevent'
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)#difference between each event time stamp and time stamp next event
    tmp = tmp.fillna(pd.NaT) #missing values for tmp set to zero = no next event --> set tmp to zero
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) #conversion to minutes   # m is for minutes

    #calculate time since start of the case --> result = new column named 'timesincecasestart'
    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1] #difference between each event's timestamp and timestamp of the last event (represents the start of the case) in the group
    tmp = tmp.fillna(pd.NaT) #missing values for tmp set to zero
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) #conversion to minutes   # m is for minutes

    #sort the group by timespamp is ascending order
    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    #assign event number to each event in the group, starting with 1 --> result= sequential order
    group["event_nr"] = range(1, len(group) + 1)

    #return the modified group
    return group

#Calculate number open cases at a given date
def get_open_cases(date):
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))
    # checks if start time each case less than/ equal to provided date
    #checks if end time each case is greater than the provided date
    #sums up cases where both conditions are true
    #dataframe dt_first_last_timestamps later defined


#own definitions, not included in original code
#definition function that transforms range values into the minimum (ex. 6-7 becomes 6)
def transform_rangevalues(value):
   # Convert the value to a string if it's a float
   if isinstance(value, float):
       value = str(value)
   if '-' in value:
       # If the value contains a range, split it and keep the first number
       return value.split("-")[0]
   else:
       # If there's no range, return the original value
       return value

#definition function that transforms scale values (ex. 2/10 becomes 2)
def transform_scalevalues(value):
   # Convert the value to a string if it's a float
   if isinstance(value, float):
       value = str(value)
   if '/' in value:
       # If the value contains a fraction, split it and keep the whole number part
       return value.split("/")[0]
   else:
       # If there's no fraction, return the original value
       return value

#definition function that rounds down decimal values (6.5 becomes 6)
def round_down(x):
   if isinstance(x, float) and not np.isnan(x):
       return math.floor(x)
   elif isinstance(x, str) and x.replace('.', '', 1).isdigit():
       # Check if string represents a number (with or without decimal point)
       return math.floor(float(x))
   else:
       return x
   
#only retain numerical values between 0-10 in a pandas dataframe column and replace the non-numeric values wiht an empty string
def retain_numeric_values(column):
    return column.apply(lambda x: x if str(x).isdigit() and 0 <= int(x) <= 10 else '')
    #lambda function checks if each value in the column is numeric (between 0-10) using the 'isdigit()' method
    #if a value is numeric it is retained, otherwise it is replaced with an empty string

#definition function converts values to lowercase
def lowercase(dataframe, columns):
    for column in columns:
            dataframe[column]= dataframe[column].str.lower()

def label_cases(group):
    if (group['hadm_id'] == 0).any():
        group[label_col] = neg_label
    else:
        group[label_col] = pos_label
    return group
