import numpy as np

def unique_sequences(traces):
    trace_tuples= trace_tuples = set(tuple(map(tuple, trace)) for trace in traces)
    return len(trace_tuples)

#statistics original BPIC dataset
#loading the datasets
traces_l_trainBPIC = np.load("C:/Users/sarah/Documents/Information management/THESIS/traces_l_train_all.npy", allow_pickle=True) #changed
traces_l_valBPIC = np.load("C:/Users/sarah/Documents/Information management/THESIS/traces_l_val_all.npy", allow_pickle=True) #changed
traces_l_testBPIC = np.load("C:/Users/sarah/Documents/Information management/THESIS/traces_l_test_all (1).npy", allow_pickle=True)

#combine arrays
all_tracesBPIC = np.concatenate((traces_l_trainBPIC, traces_l_testBPIC, traces_l_valBPIC), axis=0)

#calculate average trace length
average_lengthBPIC = np.mean([len(trace) for trace in all_tracesBPIC])
print("average trace_len BPIC2017 :", average_lengthBPIC)

#calculate number of unique traces
number_unique_tracesBPIC=unique_sequences(all_tracesBPIC)
print("number of unique tracesBPIC", number_unique_tracesBPIC)

#calculate maximum trace length
max_trace_len_allBPIC = max(
    max([len(trace) for trace in traces_l_trainBPIC]),
    max([len(trace) for trace in traces_l_testBPIC]),
    max([len(trace) for trace in traces_l_valBPIC]))
print("maximum trace lenght BPIC",max_trace_len_allBPIC)

#statistics MIMICEL dataset
#loading the dataset
traces_trainMIMICEL = np.load('C:/Users/sarah/Documents/Information management/THESIS/train1708.npy', allow_pickle=True)
traces_testMIMICEL = np.load('C:/Users/sarah/Documents/Information management/THESIS/test1708.npy', allow_pickle=True)
traces_validationMIMICEL = np.load('C:/Users/sarah/Documents/Information management/THESIS/validation1708.npy', allow_pickle=True)

#combine arrays
all_tracesMIMICEL = np.concatenate((traces_trainMIMICEL, traces_testMIMICEL, traces_validationMIMICEL), axis=0)
#calculate average trace length
average_lengthMIMICEL=np.mean([len(trace) for trace in all_tracesMIMICEL])
print("average trace_len MIMICEL",average_lengthMIMICEL)

#calculate number of unique traces
number_unique_tracesMIMCEL=unique_sequences(all_tracesMIMICEL)
print("number of uniqueMIMICEL",number_unique_tracesMIMCEL)

#calculate maximum trace length
max_trace_len_allMIMICEL = max(
    max([len(trace) for trace in traces_trainMIMICEL]),
    max([len(trace) for trace in traces_testMIMICEL]),
    max([len(trace) for trace in traces_validationMIMICEL]))
print("maximum trace length MIMICEL", max_trace_len_allMIMICEL)






