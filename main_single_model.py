
'''load package'''
from torch.utils.data import DataLoader

'''set local path for system'''
import sys
sys.path.append('')



'''load local package'''
from data_preprocess.raw2samples import raw2samples_cail_train, raw2samples_cail_test
from data_preprocess.samples2ds import convert_samples2ds_cail
from models.bert_crf_model import BertCrf

from toolbox.args_setting import initialize_args


if __name__ == '__main__':

    args = initialize_args()

    train_data_file = ''
    test_data_file = ''

    train_samples = raw2samples_cail_train(train_data_file)
    test_samples = raw2samples_cail_test(test_data_file)

    train_ds = convert_samples2ds_cail(train_samples)  # ds is abbrevation for dataset
    test_ds = convert_samples2ds_cail(test_samples)

    # update args according to tasks

    train_dl = DataLoader(train_ds, batch_size=args.trn_batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False)

    # initialize model
    bert_crf = BertCrf()






