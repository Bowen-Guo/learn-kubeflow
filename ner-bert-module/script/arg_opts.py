import argparse


def preprocess_opts():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--input_data_frame_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data frame path. Should be the .parquet file for the task.")
    parser.add_argument("--output_feature_dir",
                        default="output",
                        type=str,
                        required=True,
                        help="The feature dir.")
    parser.add_argument("--bert_model",
                        default="bert-base-cased",
                        type=str,
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    return parser


def train_opts():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_feature_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The train feature dir. Should be the data frame files for the task.")
    parser.add_argument("--bert_model",
                        default="bert-base-cased",
                        type=str,
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        default=0,
                        type=float,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    return parser


def score_opts():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--test_feature_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The test feature dir. Should be the .parquet files for the task.")
    parser.add_argument("--trained_model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="trained model folder")
    # Other parameters
    parser.add_argument("--output_eval_dir",
                        default=None,
                        type=str,
                        help="The output directory where evaluation results will be written.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--test_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")

    return parser
