import pandas as pd
from sklearn.neighbors import KernelDensity
from utils1808 import build_stateless_lstm, init_first_pop, get_x_y, transform_prefix, run_GA_loop, count_freq, merge_dicts
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, f1_score
import numpy as np
import time

#Loading datasets
traces_train = np.load('C:/Users/sarah/Documents/Information management/THESIS/train1708.npy', allow_pickle=True)
traces_test = np.load('C:/Users/sarah/Documents/Information management/THESIS/test1708.npy', allow_pickle=True)
traces_validation = np.load('C:/Users/sarah/Documents/Information management/THESIS/validation1708.npy', allow_pickle=True)


#changed from excel list to manuel list and then converted to npy:
features_list = ["subject_id", "acuity", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain",
                 "gsn", "ndc", "etccode", "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", 
                 "timesincecasestart", "event_nr", "open_cases", "gender", "race", "chiefcomplaint", "activity", 
                 "arrival_transport", "disposition", "icd_title", "rhythm", "name", "label"]

np.save('features_list.npy', features_list)
featurelist_npy = np.load('features_list.npy', allow_pickle=True).tolist()

# **Ensure features_list length matches n_features**
n_features = traces_train[0].shape[1]  # Setting n_features based on dataset
assert len(features_list) == n_features, f"Feature list length {len(features_list)} does not match n_features {n_features}"

# Initiate kde for mutation
# Fit the numeric attributes = finding mathematical model that represents the relationship between numerical variables
#vertically stacking arrays
sample= [np.max(p,axis=0)[:20] for p in np.concatenate([traces_train, traces_test,traces_validation])] #mimicel processed data: 20 numerical variables
#reshape
sample = np.array(sample).reshape(-1,20)
#fit estimated probability function to given data
kde = KernelDensity(bandwidth=0.01, kernel='gaussian').fit(sample)

# Prepare black box
min_inp_len = 5
batch_size = 32

# set this to None if your black box has non-binary target y
binary_y_index = True #mimicel label: hospitalized or not
# change the corresponding y shape: how many classes you predict
if binary_y_index:
    output_shape = 1 #mimicel 1: hospitalized or not
else:
    output_shape = 3 #BPIC2017: 4 potential outcomes: accepted/ cancelled/ denied/ pending --> predict 3 classes (variables: Concept:name_A_pending, Concept:name_A_Denied, Concept:name_A_Cancelled)

y_start_index = -output_shape  #:how many classes you predict (1: hospitalized or not)
#mimicel: y_start_index= -1 = the last column indicates the target class
#BPIC2017: y_start_index= -3 = the last 3 columns indicate the target classes

#Compute maximum length of traces across the 3 datasets
max_trace_len_all = max(
    max([len(trace) for trace in traces_train]),
    max([len(trace) for trace in traces_test]),
    max([len(trace) for trace in traces_validation]))
#compute number of features in each trace of the training dataset (it is the same number for the test and validation set)
n_feature=traces_train[0].shape[1] #should be equal to 30
model = build_stateless_lstm(input_window_len=max_trace_len_all, n_feature=n_feature,
                             output_shape=output_shape, dropout=0.2,
                             neurons_per_layer=[64, 64], masking=True)

#load the weights of the trained model and use it to make predictions for new data
model.load_weights('C:/Users/sarah/Documents/Information management/THESIS/binary_allCORRECT.keras')

#Genetic algorithm
cat_cols_start_ind = 20 #21 numerical variables, followed by categorical variables

#mapping features into pairs indicating a range of feature indices
#based on conclusion BPIC2017 dataset: #all numerical variables individually listed and the one_hot encoded variables are combined for each individual categorical variable
#mimicel: only 1 variable for each categorical variable --> list all variables individually
features_map = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                [6, 7], [7,8], [8,9], [9,10], [10,11], [11,12], [12,13],
                [13,14], [14,15], [15,16], [16,17],[17,18], [18,19], [19,20],
                [20, 21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27],
                [27,28], [28,29], [29, None]]

#mapping features into pairs indicating a range of feature indices
#based on conclusion BPIC2017 dataset: all numerical variables combined and the one_hot encoded variables are combined for each individual categorical variable
#mimicel: only 1 variable for each categorical variable --> list all categorical variables individually and combine the numerical variables
features_map_mut = [[0,20], [20,21],[21,22], [22,23], [23,24], [24,25], [25,26], [26,27],
                [27,28], [28,29],  [29, None]]
pc = 0.7 #crossover probability = probability with which two individuals (parents) are selected to undergo crossover (= recombination) to produce an offspring
pm = 0.2 #mutation probability= probability with which an individual will undergo mutation (= small random changes to the individual)
n_generations = 15
target_population = 600
max_ED = 0
seq_cols_start_ind = 2
inp_len = 5

#all input sequences (x) are of the same length and extract the corresponding target variables (y)
x, y = get_x_y(traces_test, inp_len=inp_len, max_trace_len_all=max_trace_len_all, y_start_index=y_start_index)

#create variables to store metrics and result during the execution of experiments
acc_list = [] #list accuracy measures
acc_1st_pop_list = [] #list accuracy measures 1st population
mcc_list = [] #list MCC (correlation coefficients): measure quality binary classification
mcc_1st_pop_list = [] #List MCC 1st population
hit_list = [] # list correct predictions
true_target_class_all = [] #list correct target class all examples
uniq_final_pop_per_class_all = [] #list unique definitive population per class
features_importance_freq = {} #dictionary frequency feature importance for classes
time_list = [] # list timestamps

# Load the feature list
features_list = np.load('features_list.npy', allow_pickle=True).tolist()

# loop: initiate first population, implementation genetic algorithm, train predictive model, evaluate model performance, saving statistics
for i, trace in enumerate(x):
    print("\nProgress:{}/{}".format(i + 1, len(x)))
    print("--Initiating first pop")
    start_time = time.time()
    flattened_1st_pop, uniq_seq, flattened_1st_pop_no_dup, y_1st_pop_no_dup = init_first_pop(prefix_of_interest=trace,
                                                                                             traces_l=traces_test,
                                                                                             max_ED=max_ED,
                                                                                             max_trace_len_all=max_trace_len_all,
                                                                                             target_population=target_population,
                                                                                             y_start_index=y_start_index,
                                                                                             seq_cols_start_ind=seq_cols_start_ind,
                                                                                             blackbox=model,
                                                                                             features_list=featurelist_npy)

    # Debugging before calling transform_prefix
    print(f"Calling transform_prefix for original prefix, trace index {i}")
    print(f"prefix shape: {trace.shape}")
    print(f"seq_cols_start_ind: {seq_cols_start_ind}")
    print(f"features_list length: {len(features_list)}")
    print(f"max_trace_len_all: {max_trace_len_all}")

    org_prefix_flat = transform_prefix(prefix=trace,
                                       seq_cols_start_ind=seq_cols_start_ind,
                                       features_list=featurelist_npy,
                                       max_trace_len_all=max_trace_len_all,
                                       restore=False)
    interpret_x = []
    true_target_class = []
    uniq_final_pop_per_class = []
    print(f"shape flattened_1st_pop", flattened_1st_pop.shape)
    print('--Starting GA_loop')
    for target_class in [0, 1]:  # for binary classification

        final_pop = run_GA_loop(n_generations=n_generations, target_population=target_population, pc=pc, pm=pm,
                                org_prefix_flat=org_prefix_flat, flattened_prefixes=flattened_1st_pop,
                                cat_cols_start_ind=cat_cols_start_ind, uniq_seq=uniq_seq, kde=kde,
                                max_trace_len_all=max_trace_len_all, seq_cols_start_ind=seq_cols_start_ind,
                                max_ED=max_ED, target_class=target_class, blackbox=model, features_map=features_map,
                                features_map_mut=features_map_mut, features_list=featurelist_npy)

        final_pop_restored = np.array([transform_prefix(n_p, seq_cols_start_ind,
                                                        featurelist_npy, max_trace_len_all, restore=True)
                                       for n_p in final_pop])
        first_event_last_variable = final_pop_restored[:, 0, -1]
        number_samples_label_0 = np.sum(first_event_last_variable == 0)
        #debugging statement
        print(number_samples_label_0)
        num_true_target_class = np.sum(np.argmax(model.predict(final_pop_restored), axis=-1) == target_class)
        uniq_final_pop = len([np.array(s) for s in set(tuple(s) for s in final_pop)])
        true_target_class.append(num_true_target_class)
        uniq_final_pop_per_class.append(uniq_final_pop)
        interpret_x.append(final_pop)

    true_target_class = np.array(true_target_class)
    uniq_final_pop_per_class = np.array(uniq_final_pop_per_class)
    interpret_x = np.array([f_p for c in interpret_x for f_p in c], dtype=object)
    np.random.shuffle(interpret_x)
    interpret_x = list(interpret_x)
    #debugging statement
    print('-- Finished GA_loop')
    
    # Define all possible labels for binary classification
    all_labels = [0, 1]
    
    # append the prefix of interest
    interpret_x.append(org_prefix_flat)
    interpret_x_restored = np.array([transform_prefix(n_p, seq_cols_start_ind,
                                                      featurelist_npy, max_trace_len_all, restore=True)
                                     for n_p in interpret_x])
    y = np.argmax(model.predict(interpret_x_restored), axis=-1)

    #changed compared to original code
    seq_cat = [k for p in interpret_x for k, v in enumerate(uniq_seq)
               if ','.join(p[seq_cols_start_ind:].astype(np.float64).astype(str)) == ','.join(v.astype(str))]

    # seq_cat_1hot = get_one_hot(seq_cat, len(uniq_seq))
    seq_freq = np.array([np.sum(r[:, cat_cols_start_ind:], axis=0) for r in interpret_x_restored])

    interpret_x_encoded = np.array([np.concatenate([p[:seq_cols_start_ind], seq_freq[i]]) 
                                for i, p in enumerate(interpret_x)])


    train_test_split = 0.2
    split_ind = -round(len(interpret_x_encoded) * train_test_split)

    x_train = interpret_x_encoded[:split_ind]
    y_train = y[:split_ind]
    x_test = interpret_x_encoded[split_ind:]
    y_test = y[split_ind:]

    clf = tree.DecisionTreeClassifier(max_depth=3, min_impurity_decrease=0.005)
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    #confusion_matrix(y_test, y_pred)
    blackbox_label = np.argmax(model.predict(trace.reshape(1, max_trace_len_all, -1)), axis=-1)[0]

    # Surrogate model prediction on the same transformed data
    y_surrogate = clf.predict(interpret_x_restored)

    #Fidelity part added, not included in the original code
    # Calculate fidelity for each trace in the loop
    fidelity = accuracy_score([blackbox_label], [y_pred[-1]])
    fidelity_list.append(fidelity)

    print(f'Fidelity for Trace {i + 1}: {fidelity:.4f}')

    # Calculate fidelity for the list of prefixes
    fidelity_prefixes = accuracy_score(y, y_surrogate)
    fidelity_list.append(fidelity_prefixes)
    # debugging statement
    print(f'Fidelity for Prefixes (Trace {i + 1}): {fidelity_prefixes:.4f}')
    
    #created variable instead of above
    conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels) #added labels variable compared to original code
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f'Confusion Matrix for Trace {i+1}:\n', conf_matrix)
    print('Blackbox Prediction:', blackbox_label)

    hit = (blackbox_label == y_pred[-1])
    print("uniq seq:", uniq_seq)

    # result_dict = {
    #     # 'uniq_seq': uniq_seq,
    #     # 'interpret_x': interpret_x,
    #     'accuracy_score': accuracy_score(y_test, y_pred),
    #     'balanced_accuracy_score': balanced_accuracy_score(y_test, y_pred),
    #     'mcc': matthews_corrcoef(y_test, y_pred),
    #     'f1_score': f1_score(y_test, y_pred, average='micro'),
    #     # 'accuracy_score_1st_pop': accuracy_score_1st_pop,
    #     # 'balanced_accuracy_score_1st_pop': balanced_accuracy_score_1st_pop,
    #     # 'mcc_1st_pop': mcc_1st_pop,
    #     # 'f1_score_1st_pop': f1_score_1st_pop,
    #     'hit': hit
    # }

    # Feature importances changed compared to original code
    importances = clf.feature_importances_
    print(f"Features list length: {len(features_list)}")
    print(f"Feature importances length: {len(importances)}")
    non_zero_importance_mask = importances > 0
    print(f"Non-zero importance mask length: {len(non_zero_importance_mask)}")

    # Check for length mismatch
    if len(non_zero_importance_mask) == len(features_list):
        important_features = np.array(features_list)[non_zero_importance_mask]
        important_importances = clf.feature_importances_[non_zero_importance_mask]
        dominant_features_dict = count_freq(important_features)
    elif len(clf.feature_importances_) < len(features_list):
        # Adjust features_list to match the feature importances length
        adjusted_features_list = features_list[:len(clf.feature_importances_)]
        important_features = np.array(adjusted_features_list)[non_zero_importance_mask]
        important_importances = clf.feature_importances_[non_zero_importance_mask]
        dominant_features_dict = count_freq(important_features)
    else:
        #debugging statement
        print("Mismatch between features list and model features.")
        dominant_features_dict = {}

    print("Important features and their importances:")
    important_features = np.array(features_list)[non_zero_importance_mask]
    important_importances = importances[non_zero_importance_mask]

    for feature, importance in zip(important_features, important_importances):
        print(f"{feature}: {importance:.4f}")
    
    acc_list.append(accuracy_score(y_test, y_pred))
    mcc_list.append(matthews_corrcoef(y_test, y_pred))

    # acc_1st_pop_list.append(accuracy_score_1st_pop)
    # mcc_1st_pop_list.append(mcc_1st_pop)

    hit_list.append(blackbox_label == y_pred[-1])
    true_target_class_all.append(true_target_class)
    uniq_final_pop_per_class_all.append(uniq_final_pop_per_class)

    print("---avg_acc:", np.mean(np.array(acc_list)))
    print("---avg_mcc:", np.mean(np.array(mcc_list)))
    print("---avg_hit:", np.mean(np.array(hit_list)))

    # CHANGED HERE ACCORDING TO BINARY CLASSES LABELING.
    # Print target class distribution
    print(f"---num_target_class, regular: {true_target_class[0]}, deviant: {true_target_class[1]}")

    # Print average target class distribution across all samples
    print("---avg_target_class:", np.mean(np.array(true_target_class_all), axis=0))

    # Print unique final population per class
    print(f"---uniq_final_pop: regular: {uniq_final_pop_per_class[0]}, deviant: {uniq_final_pop_per_class[1]}")

    # Print average unique final population per class across all samples
    print("---avg_uniq_final_pop_per_class:", np.mean(np.array(uniq_final_pop_per_class_all), axis=0))

    # Save the results to .npy files
    np.save(model.load_weights(
        'C:/Users/sarah/Documents/Information management/THESIS/eval_results/acc_array_{}.npy'.format(inp_len),
        np.array(acc_list)))
    np.save('C:/Users/sarah/Documents/Information management/THESIS/eval_results/mcc_array_{}.npy'.format(inp_len),
            np.array(mcc_list))
    np.save('C:/Users/sarah/Documents/Information management/THESIS/eval_results/hit_array_{}.npy'.format(inp_len),
            np.array(hit_list))
    np.save('C:/Users/sarah/Documents/Information management/THESIS/eval_results/true_target_class_all_{}.npy'.format(
        inp_len), np.array(true_target_class_all))
    np.save(
        'C:/Users/sarah/Documents/Information management/THESIS/eval_results/uniq_final_pop_per_class_all_{}.npy'.format(
            inp_len), np.array(uniq_final_pop_per_class_all))

    # Initialize the dictionaries
    features_importance_freq = {}  # or any initial value you need
    features_importance_freq_regular = {}  # Initialize to an empty dictionary
    features_importance_freq_deviant = {}  # Initialize to an empty dictionary

    print("uniq seq:", uniq_seq)
    # Track the most decisive features
    features_list_DT = list(featurelist_npy)  # + ['seq_var_{}'.format(i) for i in range(len(uniq_seq))]
    # features_list_DT = [f for f in features_list_DT if f != 'case:ApplicationType_New credit']

    # Update the feature importance frequency
    dominant_features_dict = count_freq(np.array(features_list_DT)[clf.feature_importances_ > 0])
    features_importance_freq = merge_dicts(features_importance_freq, dominant_features_dict)
    np.save('C:/Users/sarah/Documents/Information management/THESIS/eval_results/features_importance_freq_{}.npy'.format(
            inp_len), features_importance_freq)

    # Save feature importance for specific labels
    if blackbox_label == 0:  # Regular
        features_importance_freq_regular = merge_dicts(features_importance_freq_regular, dominant_features_dict)
        np.save(            'C:/Users/sarah/Documents/Information management/THESIS/eval_results/features_importance_freq_regular_{}.npy'.format(
                inp_len), features_importance_freq_regular)
    elif blackbox_label == 1:  # Deviant
        features_importance_freq_deviant = merge_dicts(features_importance_freq_deviant, dominant_features_dict)
        np.save('C:/Users/sarah/Documents/Information management/THESIS/eval_results/features_importance_freq_deviant_{}.npy'.format(
                inp_len), features_importance_freq_deviant)
    # Print the overall feature importance frequency
    print("---features_importance_freq:", features_importance_freq)

    # Track the elapsed time
    time_list.append((time.time() - start_time))
    print("--seconds:", (time.time() - start_time))
    print("--avg_seconds:", np.mean(time_list))


