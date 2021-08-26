import argparse

def initialize_args():
    parser = argparse.ArgumentParser()
    # requirement arg
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--max_grad_norm", default=3.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=1992, help="random seed for initialization")

    # data_preprocess阶段的setting
    parser.add_argument("--trn_batch_size", default=32, type=int,
                        help="training batch size.")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="testing batch size.")
    parser.add_argument("--max_length", default=256, type=int,
                        help="max length for input sentences.")


    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam in bert layers.")
    parser.add_argument("--crf_learning_rate", default=5e-2, type=float,
                        help="The initial learning rate for Adam in crf layers.")
    parser.add_argument("--weight_decay", default=1e-3, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--warmup_steps", default=200, type=float,
                        help="warming steps while performing linear learning rate")
    parser.add_argument("--training_steps", default=1000, type=float,
                        help="total training steps for training, usually it needs to be reset case by case")



    args = parser.parse_args(args=['--output_dir', './'])
    return args
