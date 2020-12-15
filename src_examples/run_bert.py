import argparse
import os
import kb_classifier, kb_ner

def main(args):

    if args.model_type=='classification':
        kb_classifier.rcml_main(args)
    elif args.model_type=='ner':
        kb_ner.rcml_main(args)
    else:
        print('please check the model type')

def make_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='./input_data/insu/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        default='/output_dir',
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--vocab_file", default='./vocab.txt', type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")

    parser.add_argument("--bert_model_path", default='./model/kb-albert-model', type=str, required=True,
                        help="Bert pre-trained model path")

    parser.add_argument("--multi_label",
                        action='store_true')

    parser.add_argument('--model_type',
                        type=str,
                        help='choose the ner or classification')
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--bert_model_name",
                        type=str)

    parser.add_argument("--config_file_name",
                        default='./model/kb-albert-model',
                        type=str)

    parser.add_argument("--do_train",
                        help="Whether to run training.",
                        action='store_true')

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")

    parser.add_argument("--do_prototype",
                        action='store_true',
                        help="Whether to run testcase on the testcase set.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=5.0,
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
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=1104,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)