import os
import sys
sys.path.append('../')
from toolbox.io_utils import load_pkl_object


def get_batch_gt_and_mp_cail(batch, pred):
    batch_size = batch.size()[0]
    orig_char_pos_list = batch['orig_char_pos_list']
    _batch_gt = []
    _batch_mp = []
    for i in range(batch_size):
        _sample_gt = []
        _sample_mp = []
        for j in range(orig_char_pos_list[i].size()):
            if orig_char_pos_list[i,j] != -1: # -1要不就是开头、结尾、要不就是[pad], 只有不是-1的，才是原来句子的元素
                _sample_gt.append(batch["label_ids"][i,j].item())
                _sample_mp.append(pred[i,j].item())
        _batch_gt.append(_sample_gt)
        _batch_mp.append(_sample_mp)

    return _batch_gt, _batch_mp


def get_batch_mp_cail(batch, pred):
    batch_size = batch.size()[0]
    orig_char_pos_list = batch['orig_char_pos_list']
    _batch_mp = []
    for i in range(batch_size):
        _sample_mp = []
        for j in range(orig_char_pos_list[i].size()):
            if orig_char_pos_list[i, j] != -1:  # -1要不就是开头、结尾、要不就是[pad], 只有不是-1的，才是原来句子的元素
                _sample_mp.append(pred[i, j].item())
        _batch_mp.append(_sample_mp)

    return _batch_mp


def decode_mp_function_cail(model_pred, id2label_mapping_file=None):
    if id2label_mapping_file is None:
        current_path = os.path.abspath(__file__)
        id2label_mapping_file = current_path + '../material/id2label_mapping.pkl'
    id2label_mapping = load_pkl_object(id2label_mapping_file)

    samples_size = len(model_pred)

    all_labels = []
    for i in range(samples_size):
        sample_labels = [id2label_mapping[label_id]
                         if label_id in id2label_mapping else id2label_mapping['O']
                           for label_id in model_pred[i]]
        all_labels.append(sample_labels)
    return all_labels













