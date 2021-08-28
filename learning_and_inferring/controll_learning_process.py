


def get_printing_info_for_training_cail(args, loss, batch):
    batch_size = batch['input_ids'].size()[0]
    return {'loss': loss/batch_size}


def get_printing_info_for_eval_cail():
    return {}


def check_model_performance_function_cail(ground_truth, model_pred):
    # ground_truth是一个list套list，内嵌的list是ner_label编码后的list
    # model_pred也是一个list套list，内嵌的list是ner_label编码后的list


    return


