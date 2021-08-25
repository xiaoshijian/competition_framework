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
from controll_learning_process import get_printing_info_for_training_cail
from controll_learning_process import check_model_performance_function_cail

def train_model(model,
                forward_function,
                get_optimizer_and_schedule,
                args,
                eval_model,
                train_dl,
                val_dl,
                get_printing_info_for_training,
                get_printing_info_for_eval,
                check_model_performance_function,


                eval_model, optimizer, lrs,
                train_dl, val_dl, device, epochs, eval_sep_step, max_patient):

    # 判断输入的参数的模式是否一致
    # 带有val_dl的模式或者是不带val_dl的模式的
    # 如果有val_dl的话，val_dl、eval_model, check_model_performance_function需要同时存在
    condition1 = ((val_dl is not None)
                  (eval_model is not None)
                  and (check_model_performance_function is not None))
    condition2 = (val_dl is None)
    assert (condition1 or condition2)


    best_score = 0
    best_model = deepcopy_state_dict_to_cpu(model)
    patient = 0

    # hist = History()

    # initialize optimizer, scheduler and scaler
    optimizer, scheduler = get_optimizer_and_schedule(args)
    scaler = GradScaler()

    pbar = tqdm(range(args.total_training_steps), desc=f"TRAIN")
    ds_iter = iter(train_dl)

    for curr_step in pbar:
        try:
            batch  = next(ds_iter)
        except:
            ds_iter = iter(train_dl)
            batch = next(ds_iter)

        with autocast():
            loss, _  = forward_function(args, model, batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # record and plot training info的这个模块需要往后再细化
        printing_info_for_training = get_printing_info_for_training(args, loss, batch)  # args相当于占位吧

        if curr_step != 0 and curr_step % args.eval_step == 0 and val_dl is not None:
            score, eval_loss = eval_model(args,
                                          model,
                                          forward_function,
                                          val_dl,
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

def eval_model(args, model, forward_function, val_dl, device):







def eval_model(args, model, forward_function, val_dl, device):
    model.eval()
    bar = tqdm(val_dl, desc=f"EVAL")
    loss_sum = 0
    samples_num = 0
    _gt = []
    _pp = []  # pp for predict_prob
    for batch in bar:
        loss, logit = forward_fct(model, batch, device)
        loss_sum += loss.item()
        samples_num += batch["labels"].size()[0]
        _gt.extend(batch["labels"].detach().cpu().tolist())
        _pp.extend(logit[:, 1].detach().cpu().tolist())
    _pl = [1 if p > pp_threshold else 0 for p in _pp]

    f1 = f1_score(y_true=_gt, y_pred=_pl)
    average_loss = loss_sum / samples_num

    return f1, average_loss


def pred_model(model, forward_fct, test_dl, device):
    model.eval()
    bar = tqdm(test_dl, desc=f"INFER")
    _pp = []  # pp for predict_prob
    for batch in bar:
        _, logit = forward_fct(model, batch, device)
        _pp.append(logit.detach().cpu())
    _pp = torch.cat(_pp)
    return _pp