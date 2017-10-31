FEATURE_LENGTH = 257 * 3
INNER_EMB_SIZE = 500 # for mem net
MODEL_CHOICE = 'ff' #  'ff' or 'mem' or 'cff' or 'nscff' or 'sfff'
CUDA = True
NUM_EPOCHS = 50

CONSTANTS = {'feature_length':FEATURE_LENGTH, 'inner_emb_size_for_mem':INNER_EMB_SIZE,
             'cuda':CUDA, 'model_choice':MODEL_CHOICE, 'num_epochs':NUM_EPOCHS}

def log_constants():
    fs = list(CONSTANTS.keys())
    fs.sort()
    for feature_name in fs:
        print(feature_name, CONSTANTS[feature_name])
