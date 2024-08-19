import os

# Define dataset configurations
dataset_confs1708 = {
    'case_id_col': {
        'mimicel': 'stay_id',
    },
    'activity_col': {
        'mimicel': 'activity',
    },
    'resource_col': {
        'mimicel': 'resource',  
    },
    'timestamp_col': {
        'mimicel': 'timestamps',
    },
    'label_col': {
        'mimicel': 'label',
    },
    'pos_label': {
        'mimicel': 'deviant',
    },
    'neg_label': {
        'mimicel': 'regular',
    },
    'dynamic_cat_cols': {
        'mimicel': ['activity', 'arrival_transport', 'disposition', 'icd_title', 'rhythm', 'name'],
    },
    'static_cat_cols': {
        'mimicel': ['gender', 'race', 'chiefcomplaint'],
    },
    'dynamic_num_cols': {
        'mimicel': ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'gsn', 'ndc', 'etccode', 'hour', 'weekday', 'month', 'timesincemidnight', 'timesincelastevent', 'timesincecasestart', 'event_nr', 'open_cases'],
    },
    'static_num_cols': {
        'mimicel': ['subject_id', 'hadm_id', 'acuity'],
    },
    'static_num_cols_no_hadm_id': {
        'mimicel':['subject_id', 'acuity'],
    },
    'filename': {
        'mimicel': os.path.join("C:/Users/sarah/Documents/Information management/THESIS/labeled_logs_csv_processed", "Preprocessed_mimic.csv")
    }
}