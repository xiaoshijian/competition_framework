import numpy as np
import copy

def split_samples_into_trn_and_vld_set(train_samples, trn_ratio=0.8):
    sample_index_list = [i for i in range(len(train_samples))]
    trn_size = int(len(train_samples) * trn_ratio)
    trn_set_idx_list = set(np.random.choice(np.array(sample_index_list),
                                        size=trn_size,
                                        replace=False).tolist())
    trn_samples = []
    vld_samples = []
    for idx in sample_index_list:
        if idx in trn_set_idx_list:
            trn_samples.append(copy.deepcopy(train_samples[idx]))
        else:
            vld_samples.append(copy.deepcopy(train_samples[idx]))

    return trn_samples, vld_samples
