

def forward_function_cail(args, model, batch):
    input_ids = batch['input_ids'].to(args.device)
    attention_mask = batch['attention_mask'].to(args.device)
    label_ids = batch['label_ids'].to(args.device)

    loss, prediction = model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             label_ids=label_ids)

    return loss, prediction

