'''
mainly focus on model learing, evaluating model performance and use model to infer

this part should be as general as possible, I don't want to change the code case by case

it can be a little change in task type by task type
'''

from torch.cuda.amp import GradScaler, autocast
from tqdm.notebook import tqdm # 因为一般都是用在jupyter notebook上的，所以使用tqdm.notebook
# from tqdm import tqdm # 当用在编辑器上时候，就用tqdm

import sys
sys.path.append('../')
from toolbox.io_utils import deepcopy_state_dict_to_cpu


def train_model(args,
                model,
                forward_function,
                get_optimizer_and_scheduler,
                train_dl,

                get_printing_info_for_training,

                val_dl,
                eval_model,
                get_batch_gt_and_mp,
                check_model_performance_function,
                ):

    # 判断输入的参数的模式是否一致
    # 带有val_dl的模式或者是不带val_dl的模式的
    # 如果有val_dl的话，val_dl、eval_model, check_model_performance_function需要同时存在
    condition1 = ((val_dl is not None)
                  and (eval_model is not None)
                  and (get_batch_gt_and_mp is not None)
                  and (check_model_performance_function is not None))
    condition2 = (val_dl is None)
    assert (condition1 or condition2)

    best_score = 0
    best_model = deepcopy_state_dict_to_cpu(model)
    patient = 0

    # hist = History()

    # initialize optimizer, scheduler and scaler
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)
    scaler = GradScaler()
    # all elements set, begin to train model

    pbar = tqdm(range(args.total_training_steps), desc=f"TRAIN")
    ds_iter = iter(train_dl)
    # print("step before training are settled")

    for curr_step in pbar:
        try:
            batch = next(ds_iter)
        except:
            ds_iter = iter(train_dl)
            batch = next(ds_iter)

        # set model to train
        model.train()

        with autocast():
            loss, _ = forward_function(args, model, batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # record and plot training info的这个模块需要往后再细化
        printing_info_for_training = get_printing_info_for_training(args, loss, batch)  # args相当于占位吧
        pbar.set_postfix(printing_info_for_training)
        if curr_step != 0 and curr_step % args.eval_step == 0 and val_dl is not None:
            score, eval_loss = eval_model(args,
                                          model,
                                          forward_function,
                                          val_dl,
                                          get_batch_gt_and_mp,
                                          check_model_performance_function)
            msg = '{} in Evaluation set: {}, loss in evaluation set: {}'
            print(msg.format(args.checking_metrics, score, eval_loss))

            if score > best_score:
                best_score = score
                best_model = deepcopy_state_dict_to_cpu(model)
                patient = 0
            else:
                patient += 1
                if patient > args.max_patient:
                    print('reach max patient, break')
                    break

    # 如果不带eval模式的话，保存最后的模型
    if val_dl is None:
        best_model = deepcopy_state_dict_to_cpu(model)
        best_score = args.default_score_for_only_train_mode # 为-1的时候，即使有问题的时候

    return best_model, best_score

def eval_model(args,
               model,
               val_dl,
               forward_function,
               get_batch_gt_and_mp,
               check_model_performance_function,
               ):
    model.eval()
    bar = tqdm(val_dl, desc=f"EVAL")
    loss_sum = 0
    _ground_truth = []
    _model_pred = []
    for batch in bar:
        loss, pred = forward_function(args, model, batch)
        batch_gt, batch_mp = get_batch_gt_and_mp(batch, pred)
        _ground_truth.extend(batch_gt)
        _model_pred.extend(batch_mp)
        loss_sum += loss
    score = check_model_performance_function(_ground_truth, _model_pred)
    loss = loss / len(_ground_truth)
    return score, loss

def model_infer(args,
                model,
                forward_function,
                test_dl,
                get_batch_mp,
                decode_mp_function,  # 把模型的输出转换成标准信息
                postprocess_function=None # 如果有后处理方法，就加入后处理方法
                ):
    model.eval()
    bar = tqdm(test_dl, desc=f"INFER")
    _model_pred = []
    for batch in bar:
        _, pred = forward_function(args, model, batch)
        batch_mp = get_batch_mp(batch, pred)
        _model_pred.extend(batch_mp)
    decoded_info = decode_mp_function(_model_pred)
    if postprocess_function:
        decoded_info = postprocess_function(decoded_info)
    return decoded_info


### only check for the contribution function for github


