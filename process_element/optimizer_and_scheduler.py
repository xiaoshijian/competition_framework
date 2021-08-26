from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR


# general elements for linear schedule
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_rate, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps)) * (1 - min_rate) + min_rate
        return max(
            min_rate, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_optimizer_and_scheduler_cail(args, model):
    no_decay = ['.bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'crf' not in n],
         'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if 'crf' in n],
         'lr': args.crf_learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'crf' not in n],
         'lr': args.learning_rate, 'weight_decay': 0}])

    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=args.num_warmup_steps,
                                                   num_training_steps=args.num_training_steps,
                                                   min_rate=args.min_rate)

    return optimizer, lr_scheduler

