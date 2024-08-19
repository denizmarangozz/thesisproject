import sys
from dataset_confs1708 import dataset_confs1708
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

class DatasetManager1708:

    #define class initializer
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        # List of required configuration attributes
        config_attributes = [
            'case_id_col', 'activity_col', 'timestamp_col',
            'label_col', 'pos_label', 'dynamic_cat_cols',
            'static_cat_cols', 'dynamic_num_cols', 'static_num_cols',
            'static_num_cols_no_hadm_id'
        ]



        # Column assignment
        self.case_id_col = dataset_confs1708['case_id_col'][self.dataset_name]
        self.activity_col = dataset_confs1708['activity_col'][self.dataset_name]
        self.timestamp_col = dataset_confs1708['timestamp_col'][self.dataset_name]
        self.label_col = dataset_confs1708['label_col'][self.dataset_name]
        self.pos_label = dataset_confs1708['pos_label'][self.dataset_name]
        self.dynamic_cat_cols = dataset_confs1708['dynamic_cat_cols'][self.dataset_name]
        self.static_cat_cols = dataset_confs1708['static_cat_cols'][self.dataset_name]
        self.dynamic_num_cols = dataset_confs1708['dynamic_num_cols'][self.dataset_name]
        self.static_num_cols = dataset_confs1708['static_num_cols'][self.dataset_name]
        self.static_num_cols_no_hadm_id=dataset_confs1708['static_num_cols_no_hadm_id'][self.dataset_name]

        self.sorting_cols = [self.timestamp_col, self.activity_col]

    #sort first by timestamp and then by activity within each timestamp

    #read the dataset based on configuration
    def read_dataset(self):
        #data type specification
        #initialize dictionary dtypes
        #data types dynamic and static categorical columns, case_id, label and timestamp column to object
        dtypes = {col: "object" for col in
                  self.dynamic_cat_cols + self.static_cat_cols + [self.case_id_col, self.label_col, self.timestamp_col]}
        #data types dynamic and static numerical columns to float
        for col in self.dynamic_num_cols + self.static_num_cols + [self.case_id_col]:
            dtypes[col] = "float"

        #read the csv file
        data = pd.read_csv(dataset_confs1708['filename'][self.dataset_name], sep=",", dtype=dtypes)
        #Convert timestamp column to datetime format
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data #returns the read dataset

    def cat_encoding (self,data):
        for col in self.dynamic_cat_cols + self.static_cat_cols + [self.label_col]:
            data[col]=data[col].astype('category').cat.codes
        return data

    def reorder_columns(self,data):
        order = [
            self.case_id_col,
            *self.static_num_cols,
            *self.dynamic_num_cols,
            *self.static_cat_cols,
            *self.dynamic_cat_cols,
            self.label_col,
            self.timestamp_col
        ]
        reordered_data=data[order]
        return reordered_data


    def split_data_strict(self, data, train_ratio, validation_ratio, split="temporal"):
        # Step 1: Sort data based on sorting columns
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')

        # Step 2: Group event log by case ID and identify start timestamps
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        # Step 3: Calculate the split indices
        total_cases = len(start_timestamps)
        train_end_idx = int(train_ratio * total_cases)
        val_end_idx = int((train_ratio + validation_ratio) * total_cases)

        # Step 4: Create lists of case IDs for train, validation, and test sets
        train_ids = list(start_timestamps[self.case_id_col])[:train_end_idx]
        validation_ids = list(start_timestamps[self.case_id_col])[train_end_idx:val_end_idx]
        test_ids = list(start_timestamps[self.case_id_col])[val_end_idx:]

        # Step 5: Filter data to create train, validation, and test sets
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True,
                                                                         kind='mergesort')
        validation = data[data[self.case_id_col].isin(validation_ids)].sort_values(self.sorting_cols, ascending=True,
                                                                                   kind='mergesort')
        test = data[data[self.case_id_col].isin(test_ids)].sort_values(self.sorting_cols, ascending=True,
                                                                       kind='mergesort')

        # Step 6: Ensure there is no overlap between training, validation, and test periods
        validation_ts = validation[self.timestamp_col].min()
        test_ts = test[self.timestamp_col].min()

        train = train[train[self.timestamp_col] < validation_ts]
        validation = validation[validation[self.timestamp_col] < test_ts]

        return train, validation, test
        # generate prefix data (each possible prefix becomes a trace)

    def split_event_log_by_case(self,data):
        case_dfs={}
        grouped=data.groupby(self.case_id_col)
        for case_id,group in grouped:
            sorted_group=group.sort_values(by=self.timestamp_col).reset_index(drop=True)
            selected_columns= self.static_num_cols_no_hadm_id+self.dynamic_num_cols+self.static_cat_cols+self.dynamic_cat_cols+[self.label_col]
            selected_data=sorted_group[selected_columns].to_numpy()
            case_dfs[case_id]=selected_data

        return case_dfs

    def dict_to_array(self, dictionary):
        values = list(dictionary.values())
        array = np.array(values, dtype=object)
        return array



