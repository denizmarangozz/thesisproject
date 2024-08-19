#loading required packages
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
import re
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

#uploading the dataset
event_logs=pd.read_csv('C:/Users/sarah/Documents/Information management/THESIS/mimicel.csv', sep=",")

#stay_id column
#calculate number unique values stay_id (should be equal to total number of cases = 425 028)
unique_stay_id=event_logs['stay_id'].nunique(dropna= False)
print(f"Number unique values in column 'stay_id': {unique_stay_id}")

#subject_id column
#calculate number unique values subject_id (should be equal to total number of patients = 205 466)
unique_subject_id=event_logs['subject_id'].nunique(dropna= False)
print(f"Number unique values in column 'subject_id': {unique_subject_id}")

#hadm_id column
filtered_data=event_logs.dropna(subset=['hadm_id'])
unique_stay_ids=set(filtered_data['stay_id'])
num_hosp_stay_ids=len(unique_stay_ids)

#activity column
#Frequencies activities
frequency_activity=event_logs['activity'].value_counts(dropna=False)
frequency_activity.plot(kind='bar')
plt.xlabel("unique values")
plt.ylabel('Frequency')
plt.title("Frequency unique values of activity column")
plt.show()
#number of cases where each actity happens
activity_cases_count = event_logs.groupby('activity')['stay_id'].nunique()
print(activity_cases_count)
activity_cases_count.plot(kind='bar')
plt.xlabel("Unique values")
plt.ylabel("Number cases")
plt.title("Number of cases each activity appears in")
plt.show()

#Gender column
#calculate number of unique values gender (should be equal to 3= 2 genders and account for missing values)
unique_gender=event_logs['gender'].nunique(dropna=False) 
print(f"Number unique values in column 'gender': {unique_gender}")

#Race column
#calculate number of unique values race (treat missing values as seperate category)
unique_race=event_logs['race'].nunique(dropna=False)
print(f"Number unique values in column 'race': {unique_race}")

#arrival_transport column
#calculate number of unique values arrival_transport (should be equal to 7= 6 categories and account for missing values)
unique_transport=event_logs['arrival_transport'].nunique(dropna=False)
print(f"Number unique values in column 'arrival_transport': {unique_transport}")
#calculate frequencies each category and visualize it
frequency_arrival_transport = event_logs['arrival_transport'].value_counts(dropna=False)
frequency_arrival_transport.plot(kind='bar')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Frequency arrival_transport categories')
plt.show()

#disposition column
#calculate number of unique values disposition (should be equal to 9= 8 categories and account for missing values)
unique_disposition=event_logs["disposition"].nunique(dropna=False)
print(f"Number unique values in column 'disposition': {unique_disposition}")
#calculate frequencies each category and visualize it
frequency_disposition = event_logs['disposition'].value_counts(dropna=False)
frequency_disposition.plot(kind='bar')
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Frequency disposition categories')
plt.show()

#icd_code column
#calculate number of unique values icd_code
unique_icd_code=event_logs["icd_code"].nunique(dropna=False)
print(f"Number unique values in column 'icd_code': {unique_icd_code}")

#icd_title column
#calculate number of unique values icd_title
unique_icd_title=event_logs["icd_title"].nunique(dropna=False)
print(f"Number unique values in column 'icd_title': {unique_icd_title}")

#comparison icd_code and icd_title
# create Crosstab between the 2 columnsn
cross_tab = pd.crosstab(event_logs['icd_code'], event_logs['icd_title'])
#execute chi test
chi2, _, _, _ = chi2_contingency(cross_tab)
#Number of rows and columsn in pivot table
n = cross_tab.sum().sum()
min_dim = min(cross_tab.shape) - 1
#calculate Cramer's V coefficient
cramers_v = np.sqrt(chi2 / (n * min_dim))
print("Cramér's V-coëfficiënt between icd_code and icd_title:", cramers_v)

#pain column
#calculate number of unique values in pain column
unique_pain=event_logs["pain"].nunique(dropna=False)
print(f"Number unique values in column 'pain': {unique_pain}")
#calculate frequency each category
frequency_pain = event_logs['pain'].value_counts(dropna=False)
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

#definition function that only keeps the integer values between zero and 10 and replace all other values with NaN
def retain_integer_values(column):
  return column.apply(lambda x: x if str(x).isdigit() and 0 <= int(x) <= 10 else '')

event_logs["pain"]=event_logs['pain'].apply(transform_rangevalues)
event_logs['pain'] = event_logs['pain'].apply(transform_scalevalues)
event_logs['pain']=event_logs['pain'].apply(round_down)
event_logs['pain']=retain_integer_values(event_logs['pain'])
frequency_pain_integer=event_logs['pain'].value_counts(dropna=False)

#chiefcomplaint column
#calculate number of unique values chiefcomplaint
unique_chiefcomplaint=event_logs["chiefcomplaint"].nunique(dropna=False)
print(f"Number unique values in column 'chiefcomplaint': {unique_chiefcomplaint}")
#calcuate frequency each category
frequency_chiefcomplaint = event_logs['chiefcomplaint'].value_counts(dropna=False)
#transformation to lower case
event_logs['chiefcomplaint']=event_logs['chiefcomplaint'].str.lower()
unique_chiefcomplaint_lower=event_logs['chiefcomplaint'].nunique(dropna=False)
frequency_chiefcomplaint_lower=event_logs['chiefcomplaint'].value_counts(dropna=False)
#threshold frequency categories = 10
categories_less_than_10=frequency_chiefcomplaint_lower[frequency_chiefcomplaint_lower<10].index #categories that have frequency less than 10
event_logs['chiefcomplaint'] = event_logs['chiefcomplaint'].apply(lambda x: x if x not in categories_less_than_10 else np.nan) #replace infrequent categories with NaN values
frequency_chiefcomplaint_lower10=event_logs['chiefcomplaint'].value_counts(dropna=False)
unique_chiefcomplaint_lower10=event_logs['chiefcomplaint'].nunique(dropna=False)

#rhythm column
#calculate number of unique values rhythm
unique_rhythm=event_logs["rhythm"].nunique(dropna=False)
print(f"Number unique values in column 'rhythm': {unique_rhythm}")

#name column
#calculate number of unique values name
unique_name=event_logs["name"].nunique(dropna=False)
print(f"Number unique values in column 'name': {unique_name}")
#calcuate frequency each category
frequency_name = event_logs['name'].value_counts(dropna=False)
#transformation to lower case
event_logs['name']=event_logs['name'].str.lower()
unique_name_lower=event_logs['name'].nunique(dropna=False)
frequency_name_lower=event_logs['name'].value_counts(dropna=False)

#etccode column
#calculate number of unique values etccode
unique_etccode=event_logs['etccode'].nunique(dropna= False)
print(f"Number unique values in column 'etccode': {unique_etccode}")
#etcdescription column
#calculate number of unique values name
unique_etcdescription=event_logs["etcdescription"].nunique(dropna=False)
print(f"Number unique values in column 'etcdescription': {unique_etcdescription}")
#comparison etccode and etcdescription column
#calculate correlation between etcdescription and etccode to determine whether they contain the same information
# create Crosstab between the 2 columnsn
cross_tab = pd.crosstab(event_logs['etccode'], event_logs['etcdescription'])
#execute chi test
chi2, _, _, _ = chi2_contingency(cross_tab)
#Number of rows and columsn in pivot table
n = cross_tab.sum().sum()
min_dim = min(cross_tab.shape) - 1
#calculate Cramer's V coefficient
cramers_v = np.sqrt(chi2 / (n * min_dim))
print("Cramér's V-coëfficiënt between etccode and etcdescription:", cramers_v)

