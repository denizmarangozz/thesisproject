import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K
from math import ceil
from sklearn.neighbors import KernelDensity
from sklearn.metrics import DistanceMetric
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the feature list manually
features_list = [
    "subject_id", "acuity", "temperature", "heartrate", "resprate",
    "o2sat", "sbp", "dbp", "pain", "gsn", "ndc", "etccode", "hour", 
    "weekday", "month", "timesincemidnight", "timesincelastevent", 
    "timesincecasestart", "event_nr", "open_cases", "gender", "race", 
    "chiefcomplaint", "activity", "arrival_transport", "disposition", 
    "icd_title", "rhythm", "name", "label"
]
np.save('features_list.npy', features_list)
featurelist_npy = np.load('features_list.npy', allow_pickle=True).tolist()

#prepare dataset for training a neural network
    #traces_l: list of traces of data
    #inp_len: length input sequences used for training
    #max_trace_len_all: maximum length traces across all data instances
    #y_start_index: specifies starting index of labels within each data instance
    #binary_y_index
def get_x_y(traces_l, inp_len, max_trace_len_all, y_start_index, binary_y_index=None):
    x_l = [t[:inp_len] for t in traces_l if len(t) > inp_len] #filter out traces that are shorter than inp_len
    x = np.array([np.pad(t, ((0, max_trace_len_all - t.shape[0]), (0, 0))) for t in x_l]) #each sequence in x_l is padded with zeros to ensure they all have the same length of max_trace_len_all
    #extracting targets and converting them into a numpy array
    if binary_y_index:
        y = np.array([t[-1][binary_y_index] for t in traces_l if len(t) > inp_len]) #extract binary target value from the last element of each trace that has more than inp_len elements
    else:
        y = np.array([t[-1][y_start_index] for t in traces_l if len(t) > inp_len]) #extract the target sequence starting from y_start_index from the last element of each trace that has more than inp_len elements

    return x, y

#Build neural network with LTSM architecture
    #input_window= maximum length of all traces in the datasets
    #n_feature= number of features each time step in input sequence has
    #output_shape= shape of model's output
    #Dropout= dropout rate applied to  hidden layers of the model (regularization technique to avoid overfitting)
    #neurons_per_layer= here the model has 2 LTSM layers, each with 64 neurons
    #masking: here used with sequential data
def build_stateless_lstm(input_window_len, n_feature, output_shape,
                         dropout=0.2, neurons_per_layer=None, masking=True):
    model = Sequential()
    
    if masking:
        model.add(Masking(mask_value=0, input_shape=(input_window_len, n_feature)))
    if neurons_per_layer == None:
        model.add(LSTM(input_window_len, input_shape=(
            input_window_len, n_feature), dropout=dropout))
    elif len(neurons_per_layer) == 1:
        model.add(LSTM(neurons_per_layer[0], input_shape=(
            input_window_len, n_feature), dropout=dropout))
    elif len(neurons_per_layer) > 1:
        for i, neurons in enumerate(neurons_per_layer):
            if i == 0:
                model.add(LSTM(neurons_per_layer[0], input_shape=(
                    input_window_len, n_feature), return_sequences=True, dropout=dropout))
            elif i == len(neurons_per_layer) - 1:
                model.add(LSTM(neurons, dropout=dropout))
            else:
                model.add(LSTM(neurons, return_sequences=True, dropout=dropout))
    model.add(Dense(output_shape, activation='sigmoid'))
    if output_shape == 1:
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    else:
        metrics = [tf.keras.metrics.CategoricalAccuracy()]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    #debugging statement to print out the model architecture and input/output shapes
    model.summary()
    
    return model

#calculate edit distance between string s and t using Levensthein distance metric
    #edit distance= minimum number of single character edits required to change one string into the other
    #s= source string
    #t=target string
    #cost (1,1,1): cost associated with deletions, insertions and substitutions
def get_ED(s, t, costs=(1, 1, 1)):
    rows = len(s) + 1
    cols = len(t) + 1
    deletes, inserts, substitutes = costs

    dist = [[0 for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes

    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts

    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + cost)  # substitution

    return dist[row][col]



#transformation prefixes of event sequences into different format, here used for sequence prediction tasks
    #own definition of function
    #prefix= input
    #features_list: containing names all features
    #max_trace_len_all: maximum length traces across all data instances

def transform_prefix(prefix, seq_cols_start_ind, features_list, max_trace_len_all, restore=False):
    num_features = len(features_list)
    if restore:
        #debugging statement
        print("restore is True")
        prefix_len = len(prefix)
        #debugging statement
        print(f"prefix shape: {prefix.shape}")
        print(f"Length of prefix: {prefix_len}")
        print(f"Number of features: {num_features}")
        
        # Check if the prefix length is divisible by the number of features
        if prefix_len % num_features != 0:
            print("Warning: Length of prefix is not divisible by the number of features.")
            padding_length = num_features - (prefix_len % num_features)
            padded_prefix = np.pad(prefix, (0, padding_length), mode='constant')
            # debugging statement
            print(f"Padded prefix length: {len(padded_prefix)}")
        else:
            padded_prefix = prefix

        # Update num_time_steps after padding
        num_time_steps = len(padded_prefix) // num_features
        print(f"Calculated number of time steps: {num_time_steps}")
        
        # Prepare the transformed_prefix array
        transformed_prefix = np.zeros((max_trace_len_all, num_features))
        
        try:
            transformed_prefix[:num_time_steps, :num_features] = padded_prefix.reshape((num_time_steps, num_features))
        except ValueError as e:
            #print(f"Error reshaping prefix: {e}")
            raise ValueError("Reshape failed due to mismatched dimensions.")
    
    else:
        #debugging statement
        print("restore is False")
        transformed_prefix = prefix.flatten()

    return transformed_prefix


#get the initial population for the genetic algorithm: find similar prefices in a set of traces based on edit distance
def get_first_pop(trace, traces_l, max_ED, max_trace_len_all, y_start_index, features_list,
                  seq_cols_start_ind, blackbox, target_label=None):
    # get the input length based on sum of each row in the trace matrix (first row in trace where the sum of the elements equals to zero)
    #consider this point in the trace as end of relevant data
    inp_len = [i for i, v in enumerate(np.sum(trace, axis=1)) if v == 0][0]

    #debugging statements
    print("input length:", inp_len)
    print(f"Shape of trace: {trace.shape}")
    print(f"Shape of traces_l: {np.array(traces_l).shape}")

    #select candidate prefixes from the list traces_l that are longer than the current input length
    #pad each selected prefix with zeros to get the same length as max_trace_len_all (=218 for MIMCEL)
    def get_candidate_prefixes(traces_l, max_trace_len_all, inp_len, blackbox):

        x_l = [t[:inp_len] for t in traces_l if len(t) > inp_len]
        x = np.array([np.pad(t, ((0, max_trace_len_all - t.shape[0]), (0, 0))) for t in x_l])
        return x

    # define min_inp_len and max_inp_len, and in case they are "at the border"
    #min and max input length adjustment
    if (inp_len - max_ED) <= 0:
        min_inp_len = 0
    else:
        min_inp_len = inp_len - max_ED

    if (inp_len + max_ED) >= max_trace_len_all:
        max_inp_len = max_trace_len_all
    else:
        max_inp_len = inp_len + max_ED

    #debugging statements
    print(f"Value min_inp_len", min_inp_len)
    print(f"Value max_inp_len", max_inp_len)
    final_prefixes = [] #list where selected prefixes are saved
    seq_l = [] #list of sequences of the selected prefixes
    trace_seq = np.argmax(trace[:, seq_cols_start_ind:], axis=1)[:inp_len]
    
    print("Comparing prefixes with length: {} to {}".format(min_inp_len, max_inp_len))
    #sequence comparison and filtering: compares sequence of the input trace with sequences of candidate prefixes: if the edit distance between them is wihtin maximum edit distance, the candidate prefix is added to final_prefixes list
    for l in range(min_inp_len, max_inp_len + 1):
        cand_prefixes = get_candidate_prefixes(traces_l=traces_l, max_trace_len_all=max_trace_len_all,
                                               inp_len=l, blackbox=blackbox)
        # Calculate edit distance
        for c in cand_prefixes:
            c_seq = np.argmax(c[:, seq_cols_start_ind:], axis=1)[:l]
            if get_ED(trace_seq, c_seq) <= max_ED:
                final_prefixes.append(c)
                seq_l.append(c_seq)

    final_prefixes = np.array(final_prefixes)
    # debugging statement
    print(f"Shape of final_prefixes before filtering: {final_prefixes.shape}")
    
    if target_label != None:
        print("targeting")
        final_prefixes = final_prefixes[blackbox.predict_classes(final_prefixes) == target_label]

    if final_prefixes.shape[0] == 0:
        org_prefix_flat = transform_prefix(prefix=trace,
                                           seq_cols_start_ind=seq_cols_start_ind,
                                           features_list=featurelist_npy,
                                           max_trace_len_all=max_trace_len_all,
                                           restore=False)
        
        print(f"Shape of org_prefix_flat: {org_prefix_flat.shape}")
        print('No similar prefixes found, consider increasing the max_ED')
        return org_prefix_flat, []

    uniq_seq = [np.array(s) for s in set(tuple(s) for s in seq_l)]

    return final_prefixes, uniq_seq

#calculate fitness of genetic algorithm based on various criteria

def calculate_fitness(org_prefix_flat, flattened_prefixes, cat_cols_start_ind, max_trace_len_all,
                      features_list, seq_cols_start_ind, max_ED, target_class, blackbox, alpha=0.8, skip_ED=True):
    
    # Ensure flattened_prefixes is a NumPy array
    # debugging statement
    print ("flattened prefixes before conversion to numpy array")
    print(flattened_prefixes)
    if isinstance(flattened_prefixes, list):
        flattened_prefixes = np.array(flattened_prefixes)
    print("flattened prefixes after conversion to numpy array")
    print (flattened_prefixes)
    num_features=len(features_list)
    
    # Define distance metrics
    jaccard_dist = DistanceMetric.get_metric('jaccard')
    euclidean_dist = DistanceMetric.get_metric('euclidean')

    # Split numeric and categorical columns
    num_cols = [f_p[:cat_cols_start_ind] for f_p in flattened_prefixes]
    num_cols.insert(0, org_prefix_flat[:cat_cols_start_ind])
    #debugging statements
    print(f"shape num_cols: {np.shape(num_cols)}")
    num_dist = euclidean_dist.pairwise(num_cols)[0, 1:]
    print(f"shape num_dist", num_dist.shape)
    
    # Ensure cat_cols is defined and populated correctly
    catenc_cols = [f_p[cat_cols_start_ind:num_features-1] for f_p in flattened_prefixes]
    catenc_cols.insert(0, org_prefix_flat[cat_cols_start_ind:num_features-1])
    # debugging statement
    print(f"shape of catenc_cols: {np.shape(catenc_cols)}")
    catenc_dist = jaccard_dist.pairwise(catenc_cols)[0, 1:]
    print(f"shape of catenc_dist", catenc_dist.shape)

    
    # Restore prefixes and get class predictions
    restored_prefixes = np.array([transform_prefix(f_p, seq_cols_start_ind, features_list,
                                                   max_trace_len_all, restore=True)
                                  for f_p in flattened_prefixes])

    # debugging statement
    print(f"Restored prefixes shape: {restored_prefixes.shape}")
    print(f"First element in restored prefixes: {restored_prefixes[0]}")

    # debugging statement
    #ensuring shape restored prefix matches the expected input shape of blackbox model
    print("Model input shape:", blackbox.input_shape)
    print("Input tensor shape:", restored_prefixes.shape)
    #ensure inputshape is correct
    expected_input_shape=blackbox.input_shape[1:] #exclude the batch dimension
    if restored_prefixes.shape[1:] != expected_input_shape:
        raise ValueError(f"Input shape mismatch: expected {expected_input_shape}, got {restored_prefixes.shape[1:]}")
    #ensure batch dimension is present
    if len(restored_prefixes.shape) == len(expected_input_shape):
        restored_prefixes = np.expand_dims(restored_prefixes, axis=0)
    #ensure restored prefix has the correct datatype
    restored_prefixes=np.array(restored_prefixes, dtype=np.float32)
    
    restored_prefixes_class = np.argmax(blackbox.predict(restored_prefixes), axis=-1)
        
    # Fitness calculation
    indicator_1 = restored_prefixes_class == target_class
    indicator_2 = np.array([np.array_equal(org_prefix_flat[:seq_cols_start_ind], f_p[:seq_cols_start_ind])
                            for f_p in flattened_prefixes])

    if skip_ED:
        fitness = indicator_1 + (1 - ((num_dist + catenc_dist) / 2)) - indicator_2
    else:
        ED = np.array([get_ED(org_prefix_flat[seq_cols_start_ind:], f_p[seq_cols_start_ind:])
                       for f_p in flattened_prefixes]) * (alpha / max_ED)
        fitness = indicator_1 + (1 - ((num_dist + catenc_dist + ED) / 3)) - indicator_2

    return fitness
    
#Implementation genetic operations
def crossover_mutate(next_pop, features_map, features_map_mut, kde, uniq_seq,max_trace_len_all, pc=0.7, pm=0.2, shuffle=True):
    #parent selection
    #select subset individual current population (next_pop) with probability pc
    #select pair of parents from selected parent indices with replacement --> 2D array where each row represents a pair of parents
    parents_size = ceil(len(next_pop) * pc)
    #debugging statements
    print(f"parents size", parents_size)
    parents_ind = np.random.choice(range(len(next_pop)), size=parents_size, replace=False)
    parents = np.array(next_pop)[np.random.choice(parents_ind, size=(parents_size * 2, 2), replace=True)]
    # uniform crossover: randomly select features from each parent
    crossovered = [np.concatenate([p[v][features_map[i][0]:features_map[i][1]]
                                   for i, v in enumerate(np.random.randint(0, 2, size=len(features_map)))])
                   for p in parents]
    crossovered = np.array(crossovered, dtype=object)
    # debugging statement
    print(f"shape crossovered", crossovered.shape)

    #own defintion mutation operation
    def swap_mutation(individual):
        indices = np.random.choice(len(individual), size=2, replace=False)
        individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]
        return individual

    def mutate(crossovered, pm, mutation_operator):
        #crossovered= population to mutate
        #pm= mutation probability
        #mutation_operator= function performing the actual mutation
        #Returns mutated population

        mutation_size = ceil(crossovered.shape[0] * pm)
        mutation_ind = np.random.choice(range(crossovered.shape[0]), size=mutation_size, replace=False)
        non_mutation_ind = [i for i in range(crossovered.shape[0]) if i not in mutation_ind]
        mutated_children = crossovered.copy()[mutation_ind]

        # Apply the mutation operator to the selected individuals
        for i in range(len(mutated_children)):
            mutated_children[i] = mutation_operator(mutated_children[i])
        #debugging statement
        print(f"shape mutated_children", mutated_children.shape)
        print(f"shape non mutated children", crossovered[non_mutation_ind].shape)
        mutated_pop = np.concatenate([mutated_children, crossovered[non_mutation_ind]])
        return mutated_pop

    new_next_pop=mutate(crossovered,pm,swap_mutation)
    # debugging statement
    print(f"shape new_next_pop", new_next_pop.shape)


    if shuffle:
        np.random.shuffle(new_next_pop)

    return new_next_pop

#initialize first population for genetic algorithm: ensures diversity by replicating and transforming the prefixes into suitable format for subsequent evolutionary operations
def init_first_pop(prefix_of_interest, traces_l, max_ED, max_trace_len_all, target_population,
                   y_start_index, seq_cols_start_ind, blackbox, features_list, return_org=True):
    #initialization prefixes: get set of candidate prefixes (prefixes) based on given prefix of interest (prefix_of_interest) + obtains set unique sequences
    prefixes, uniq_seq = get_first_pop(trace=prefix_of_interest,
                                       traces_l=traces_l,
                                       max_ED=max_ED,
                                       max_trace_len_all=max_trace_len_all,
                                       y_start_index=y_start_index,
                                       features_list=featurelist_npy,
                                       seq_cols_start_ind=seq_cols_start_ind,
                                       blackbox=blackbox,
                                       target_label=None)
    #population expansion: replicates obtained prefixes to create initial population (init_pop) with target size (target_population)
        #ceil(target_population * 2 / prefixes.shape[0])
        #Target population = desired size final population (set to 600)
        #prefixes.shape[0]: number of rows in the prefixes array (= number of traces)
        #np.tile (prefixes, (n,1,1,)): prefixes will be repeated n times among the first axis
    #Result: creates an initial population by repeating the prefixes array enough times to meet/ exceed "target_population*2"
    #here target_population= 600 and prefixes.shape[0]= 7364 so no repition needed: init_pop identical to prefixes
    init_pop = np.tile(prefixes, (ceil(target_population * 2 / prefixes.shape[0]), 1, 1))
    #Flattening and transformation: for each prefix in the initial populatin, it flattens the prefix and transforms it into a format suitable for further processing --> stored in flattened_first_pop
    flattened_first_pop = np.array([transform_prefix(p,
                                                     seq_cols_start_ind=seq_cols_start_ind,
                                                     features_list=featurelist_npy,
                                                     max_trace_len_all=max_trace_len_all,
                                                     restore=False)
                                    for p in init_pop])
    #debugging statements
    print ("flattened_first_pop")
    print(flattened_first_pop)
    print ("shape flattened_first_pop")
    print(flattened_first_pop.shape)
    #optional return
    if return_org:
        y_first_pop_no_dup = np.argmax(blackbox.predict(prefixes), axis=-1)
        flattened_first_pop_no_dup = [transform_prefix(p,
                                                       seq_cols_start_ind=seq_cols_start_ind,
                                                       features_list=featurelist_npy,
                                                       max_trace_len_all=max_trace_len_all,
                                                       restore=False)
                                      for p in prefixes]
        return flattened_first_pop, uniq_seq, flattened_first_pop_no_dup, y_first_pop_no_dup
    else:
        return flattened_first_pop, uniq_seq



#implementation genetic algorithm
def run_GA_loop(n_generations, target_population, pc, pm, org_prefix_flat, flattened_prefixes, features_list,
                cat_cols_start_ind, seq_cols_start_ind, max_ED, target_class, blackbox,
                features_map, features_map_mut, uniq_seq, kde, max_trace_len_all, skip_ED=True):
    #initialization middle population
    middle_pop_size = round(target_population * 1.5)
    #generational loop
    for i in range(n_generations):
        print(i, 'generation')
        # Selection
        #calculate fitness individual in the population
        #debugging statements
        print(f"shape org_prefix_flat before fitness calculation", org_prefix_flat.shape)
        print(f"shape flattened prefix before fitness calculation", flattened_prefixes.shape)
        fitness = calculate_fitness(org_prefix_flat=org_prefix_flat,
                                    flattened_prefixes=flattened_prefixes,
                                    cat_cols_start_ind=cat_cols_start_ind,
                                    max_trace_len_all=max_trace_len_all,
                                    features_list=featurelist_npy,
                                    seq_cols_start_ind=seq_cols_start_ind,
                                    max_ED=max_ED,
                                    target_class=target_class,
                                    blackbox=blackbox,
                                    skip_ED=skip_ED,
                                    )
        #select top individuals from population based on their fitness value
        fitness_cutoff = np.sort(fitness)[-middle_pop_size]
        next_pop_ind = fitness >= fitness_cutoff

        next_pop = np.array([next_c for i, next_c in enumerate(flattened_prefixes) if next_pop_ind[i]], dtype=object)
        next_pop = np.array([np.array(s) for s in set(tuple(s) for s in next_pop)], dtype=object)
        #if number of selected individuals exceeds middle_pop_size: it truncates the population to middle_pop_size
        if len(next_pop) > middle_pop_size:
            next_pop = next_pop[: middle_pop_size]

        # Using sets to handle unique entries instead of np.unique
        unique_next_pop = set(tuple(n_p) for n_p in next_pop)
        #debugging statement
        print("unique next pop:", len(unique_next_pop))

        # Print unique next population by truncating to first 3 columns
        unique_next_pop_no_seq = set(tuple(n_p[:3]) for n_p in next_pop)
        #debugging statement
        print("unique_next_pop_no_seq:", len(unique_next_pop_no_seq))
        print("unique_fitness:", len(np.unique(fitness)))
        print("avg_fitness:", np.mean(fitness))

        #population maintenance: if the size of selected population (next_pop) is less than the target population size, it randomly duplicates individuals unitl the target size is reached
        if len(next_pop) < target_population:
            next_pop = np.concatenate([next_pop,
                                       next_pop[np.random.randint(len(next_pop),
                                                                  size=target_population - len(next_pop))]])
        # Crossover&Mutation
        next_pop_before_selection = crossover_mutate(next_pop=next_pop,
                                                     features_map=features_map,
                                                     features_map_mut=features_map_mut,
                                                     kde=kde,
                                                     uniq_seq=uniq_seq,
                                                     max_trace_len_all=max_trace_len_all,
                                                     pc=pc,
                                                     pm=pm,
                                                     shuffle=True,
                                                     )
        #debugging statement
        print(f"shape next_pop after crossover and mutation")
        print(next_pop_before_selection.shape)
        flattened_prefixes = next_pop_before_selection
        #debugging statement
        print(f"shape next pop", next_pop.shape)
        print("next_pop:", next_pop.shape[0])
        print("next_pop_before_selection: ", next_pop_before_selection.shape[0])
        #finalization
        #if it's the largest population of if the average fitness value exceeds a treshold it performs a final fitness evaluation and selects the final population
        if (((i + 1) == n_generations) or np.mean(fitness) >= 1.5):
            # if (((i + 1) == n_generations)):
            flattened_prefixes = np.array([np.array(s) for s in set(tuple(s) for s in flattened_prefixes)],
                                          dtype=object)
            # debugging statement
            print(f"next population after selection")
            print(flattened_prefixes.shape)
            fitness = calculate_fitness(org_prefix_flat=org_prefix_flat,
                                        flattened_prefixes=flattened_prefixes,
                                        cat_cols_start_ind=cat_cols_start_ind,
                                        max_trace_len_all=max_trace_len_all,
                                        features_list=featurelist_npy,
                                        seq_cols_start_ind=seq_cols_start_ind,
                                        max_ED=max_ED,
                                        target_class=target_class,
                                        blackbox=blackbox,
                                        skip_ED=skip_ED)
            #ensures final popuulation size does not excees the target population size
            if len(fitness) > target_population:
                # debugging statement
                print(f"len fitness larger than target population")
                fitness_cutoff = np.sort(fitness)[-round(target_population)]
                final_pop_ind = fitness >= fitness_cutoff
                final_pop = np.array([next_c for i, next_c in enumerate(flattened_prefixes) if final_pop_ind[i]],
                                     dtype=object)
                if len(final_pop) > target_population:
                    final_pop = final_pop[:target_population]
                    # debugging statement
                print("---unique final pop:", len([np.array(s) for s in set(tuple(s) for s in final_pop)]))
                return final_pop
            else:
                # debugging statement
                print(f"len fitness smaller than target population")
                return flattened_prefixes

def count_freq(my_list):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return freq

#combine 2 dictionaries
def merge_dicts(dict_1, dict_2):
    new_dict = {}

    same_keys = [k for k in dict_1.keys() if k in dict_2.keys()]
    uniq_keys_dict_1 = [k for k in dict_1.keys() if k not in dict_2.keys()]
    uniq_keys_dict_2 = [k for k in dict_2.keys() if k not in dict_1.keys()]
    all_keys_uniq = list((set(list(dict_1.keys()) + list(dict_2.keys()))))

    for k in all_keys_uniq:
        if k in same_keys:
            new_dict[k] = dict_1[k] + dict_2[k]
        elif k in uniq_keys_dict_1:
            new_dict[k] = dict_1[k]
        elif k in uniq_keys_dict_2:
            new_dict[k] = dict_2[k]

    return new_dict


    """
Generator function to yield batches of input-output pairs for training the LSTM model. 

Parameters:
    - traces: List of numpy arrays, where each array represents a trace with variable length.
    - y_start_index: The index where the output (y) starts in each trace.
    - max_trace_len_all: The maximum length of all traces combined.
    - binary_y_index: If not None, specifies the index for binary classification target y.
    - batch_size: Number of samples per batch.
    - min_inp_len: Minimum length of input sequence.
    - shuffle: Whether to shuffle the data before each epoch.

Yields:
    - batch_x (padded): Numpy array of shape (batch_size, max_trace_len_all, n_features) representing input sequences.
    - batch_y: Numpy array of shape (batch_size, output_shape) representing target outputs.
    """

def generator(data, y_start_index, max_trace_len_all, binary_y_index, batch_size, min_inp_len,
              shuffle=True):
    n_samples = len(data) #determines number of samples in the data
    while True:  # Make sure the generator is infinite: ensures the generator can continuously produce batches
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size): #iterate through shuffles indices in steps of batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            excerpt = indices[start_idx:end_idx] #indices current batch
            X_batch = [] #initialize empty X_batch for current batch
            y_batch = [] #initialize empty y_batch for current batch

            for i in excerpt:
                x_item = data[i][:, :][:max_trace_len_all] # extracting value for each cell i (= sequence events 1 case): select all the columns and truncate the rows to max_trace_len_all
                X_batch.append(x_item) #add x_item to X_batch

                if binary_y_index is not None:
                    y_item = data[i][:, y_start_index:][0] #extract binary target variable = extract target value for each cell i (= sequence events 1 case): select the last column = target variable
                else:
                    y_item = data[i][:, y_start_index:][0] #extract target variables starting from y_start_index (negative value= only select the last "y_start_index" columns)
                y_batch.append(y_item) #add extracted target variable(s) to y_batch

            # Pad sequences to the same length
            X_batch_padded = pad_sequences(X_batch, maxlen=max_trace_len_all, padding='post', truncating='post')

            # Convert y_batch to numpy array
            y_batch = np.array(y_batch)

            yield X_batch_padded, y_batch




#definition get_n_steps: calculate number of steps (batches) needed for one epoch of training/ validation given a dataset and a batch size
    #data: used dataset
    #batch_size
    #min_inp_len= None: only consider examples whose length is greater than or equal to minimum input length
    #returns number of steps per epoch
def get_n_steps(data, batch_size, min_inp_len=None):
    n_samples = len(data)
    print(n_samples)
    if min_inp_len is not None:
        n_samples = sum(1 for example in data if len(example) >= min_inp_len)
    steps_per_epoch = n_samples // batch_size
    if n_samples % batch_size != 0:
        steps_per_epoch -= 1
    return steps_per_epoch