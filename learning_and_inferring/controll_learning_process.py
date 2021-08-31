import os
import sys
sys.path.append('../')
from toolbox.io_utils import load_pkl_object
from toolbox.ner_utils import SeqEntityScore


def get_printing_info_for_training_cail(args, loss, batch):
    batch_size = batch['input_ids'].size()[0]
    return {'loss': loss/batch_size}


def get_printing_info_for_eval_cail():
    return {}


def check_model_performance_function_cail(ground_truth, model_pred, id2label_mapping_file=None):
    if id2label_mapping_file is None:
        current_path = os.path.abspath(__file__)
        id2label_mapping_file = current_path + '../material/id2label_mapping.pkl'
    id2label_mapping = load_pkl_object(id2label_mapping_file)
    seq_entity_score = SeqEntityScore(id2label=id2label_mapping)
    seq_entity_score.update(label_paths=ground_truth,
                            pred_paths=model_pred)

    # ground_truth是一个list套list，内嵌的list是ner_label编码后的list
    # model_pred也是一个list套list，内嵌的list是ner_label编码后的list
    overall_performance_dict, class_info = seq_entity_score.result
    acc, recall, f1 = seq_entity_score.result[0]

    return f1


