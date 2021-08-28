
'''load package'''
import torch
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

'''set local path for system'''
import sys
sys.path.append('../')

'''load local package'''
from data_preprocess.raw2samples import raw2samples_cail_train #, raw2samples_cail_test
from data_preprocess.samples2ds import CailNerDataset

from models.bert_crf_model import BertCrf
from process_element.forward_function import forward_function_cail
from process_element.optimizer_and_scheduler import get_optimizer_and_scheduler_cail

from toolbox.args_setting import initialize_args
from toolbox.setting_utils import seed_everything
from toolbox.util import split_samples_into_trn_and_vld_set

from learning_and_inferring.engine import train_model, eval_model

# from controll_learning_process import get_printing_info_for_training_cail
# from controll_learning_process import check_model_performance_function_cail


if __name__ == '__main__':

    args = initialize_args()

    train_data_file = '/Users/yuanmou006/PycharmProjects/cail2021/data/xxcq_small.json'


    # 我是不是缺了一个转换test_samples的啊，是的，但是现在没有test的
    train_samples, lim, ilm, com, ocm = raw2samples_cail_train(train_data_file)

    trn_samples, vld_samples = split_samples_into_trn_and_vld_set(train_samples)

    # update args according to tasks
    args.num_labels = len(lim)
    args.backbone_drop_rate = 0.3
    args.max_length = 256
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_performance_record_file = './model_performance_record.txt'
    args.output_file = './output_file_ner.txt'

    args.eval_step = 100
    args.total_training_steps = 1000
    args.max_patient = 5
    args.checking_metrics = "F1 score"

    args.default_score_for_only_train_mode = -1

    # prepare dataset and dataloader
    trn_ds = CailNerDataset(samples=trn_samples,
                            labels_ids_mapping=lim,
                            max_length=args.max_length)  # ds is abbrevation for dataset

    trn_dl = DataLoader(trn_ds, batch_size=args.trn_batch_size, shuffle=True)

    vld_ds = CailNerDataset(vld_samples,
                            labels_ids_mapping=lim,
                            max_length=args.max_length)

    vld_dl = DataLoader(vld_ds, batch_size=args.trn_batch_size, shuffle=False)

    # initialize model
    # bert_crf = BertCrf()
    model = BertCrf(BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext'),
                    args=args)

    seed_everything(args.seed)

    train_model(args=args,
                model=model,
                forward_function=forward_function_cail,
                get_optimizer_and_schedule=get_optimizer_and_scheduler_cail,
                train_dl=trn_dl,
                get_printing_info_for_training,
                val_dl=vld_dl,
                eval_model=None,
                check_model_performance_function=None,
                )













