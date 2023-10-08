import pickle

class DefaultConfig(object):
    
    dataset_name = 'dbd5' # set to 'masif' or 'dockground' for evaluating on these datasets
    language_model = 'ProtTrans_Bert' # can be replaced by 'ProtXLNet'

    STANDARDIZE_EDGE_FEATURES = True
    STANDARDIZE_NODE_FEATURES = True

    max_sequence_length = 1000
    neighbourhood_size = 21
    regenerate_train_val_split = True
    
    batch_size = 32 * 3
    lr = 0.01 
    weight_decay = 0.0001
    USE_STEP_LR = True
    WEIGHTED_LOSS = True
    
    dropout = 0.2
    num_workers = 1
    
    with open ("inputs/{}/dataset_sizes.pkl".format(dataset_name), "rb") as file:
        dataset_sizes = pickle.load(file)
        train_size, val_size, test_size = dataset_sizes.values()
    
    total_proteins = train_size + val_size + test_size
    total_experiments_num = 1
    total_epochs = 0 # set to 0 for only testing purpose
    
    patience_before_dropping_training = 50

    init_experiment_number = 0

    TRAIN_FOR_SINGLE_PROTEINS = True
    TRAIN_FOR_PAIRWISE_PROTEINS = True

    num_heads = 4
    num_gat_layers = 2
    pos_neg_ratio = 10
    
    print_freq = 1

    num_phychem_features = 14
    num_bert_features = 1024
    num_residue_accessibility_features = 2

    use_phychem_data = (num_phychem_features > 0)
    use_bert_data = (num_bert_features > 0)
    use_residue_accessibility_features = (num_residue_accessibility_features > 0)
    PREDICT_INTERFACE_REGION = True
    
    assert(use_bert_data or use_phychem_data)  # must have at least one

    SAVE_TEST_OUTPUTS = True
    SAVE_MODEL = True


    



