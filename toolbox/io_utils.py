from copy import deepcopy
import json
import pickle

def deepcopy_state_dict_to_cpu(model):
    output = {}
    for k, v in model.state_dict().items():
        output[k] = deepcopy(v.cpu())
    return output

def load_json_object(file):
    with open(file, 'r') as f:
        json_obj = json.load(f)
    return json_obj

def load_pkl_object(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj