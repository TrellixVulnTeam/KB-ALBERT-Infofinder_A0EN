"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn import metrics
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import MultiLabelClassification

### kyoungman.bae @ 19-05-28 
from pytorch_pretrained_bert.tokenization_morp import BertTokenizer
from tokenization_kbalbert import KbAlbertCharTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from preprocess_function import accuracy, MultiClassProcessor, MultiLabelProcessor, convert_examples_to_features, hamming_score
from transformers import AlbertForSequenceClassification, AlbertConfig

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def rcml_main(args):
    logger.info('KB-ALBERT 조항 분류기 동작')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.multi_label:
        processor = MultiLabelProcessor()
    else:
        processor = MultiClassProcessor()

    converter = convert_examples_to_features

    label_list = processor.get_labels(args.data_dir)
    label_map = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = KbAlbertCharTokenizer.from_pretrained(args.bert_model_path)

    train_sen_examples = None
    eval_sen_examples = None
    test_sen_examples = None

    num_train_optimization_steps = None

    if args.do_train:
        train_sen_examples = processor.get_train_examples(args.data_dir)
        eval_sen_examples = processor.get_dev_examples(args.data_dir)

        train_sen_features = converter(train_sen_examples, label_list, args.max_seq_length, tokenizer, args.multi_label)
        eval_sen_features = converter(eval_sen_examples, label_list, args.max_seq_length, tokenizer, args.multi_label)

        num_train_optimization_steps = int(
            len(train_sen_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.do_test:
        if args.do_prototype :
            test_sen_examples = processor.get_prototype_examples(args.data_dir)
        else :
            test_sen_examples = processor.get_test_examples(args.data_dir)

        test_sen_features = converter(test_sen_examples, label_list, args.max_seq_length, tokenizer, args.multi_label)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    config = AlbertConfig.from_pretrained(args.config_file_name, num_labels=num_labels, id2label=label_map)

    if args.do_train:
        if args.multi_label:
            model = MultiLabelClassification.from_pretrained(args.bert_model_path,
                                                             args.bert_model_name,
                                                             cache_dir=cache_dir,
                                                             num_labels=num_labels)
        else:
            model = AlbertForSequenceClassification.from_pretrained(args.bert_model_path,
                                                                    config=config)

    elif args.do_test:
        model = torch.load(os.path.join(args.bert_model_path, args.bert_model_name))

    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.do_train:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    ##train_model
    global_step = 0
    if args.do_train:
        # model.unfreeze_bert_encoder()

        if len(train_sen_features) == 0:
            logger.info("The number of train_features is zero. Please check the tokenization. ")
            sys.exit()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_sen_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_sen_input_ids = torch.tensor([f.input_ids for f in train_sen_features], dtype=torch.long)
        train_sen_input_mask = torch.tensor([f.input_mask for f in train_sen_features], dtype=torch.long)
        train_sen_segment_ids = torch.tensor([f.segment_ids for f in train_sen_features], dtype=torch.long)
        train_sen_label_ids = torch.tensor([f.label_id for f in train_sen_features], dtype=torch.long)

        eval_sen_input_ids = torch.tensor([f.input_ids for f in eval_sen_features], dtype=torch.long)
        eval_sen_input_mask = torch.tensor([f.input_mask for f in eval_sen_features], dtype=torch.long)
        eval_sen_segment_ids = torch.tensor([f.segment_ids for f in eval_sen_features], dtype=torch.long)
        eval_sen_label_ids = torch.tensor([f.label_id for f in eval_sen_features], dtype=torch.long)

        train_sen_data = TensorDataset(train_sen_input_ids, train_sen_input_mask, train_sen_segment_ids, train_sen_label_ids)
        eval_sen_data = TensorDataset(eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_sen_label_ids)

        train_sen_dataloader = DataLoader(train_sen_data, batch_size=args.train_batch_size,
                                          worker_init_fn=lambda _: np.random.seed())
        eval_sen_dataloader = DataLoader(eval_sen_data, batch_size=args.train_batch_size)

        train_loss_values, valid_loss_values = [], []
        train_acc, valid_acc = [], []
        train_f1, valid_f1 = [], []

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0

            total_loss = 0
            tr_predicted_labels, tr_target_labels = list(), list()

            for step, train_sen_batch in enumerate(tqdm(train_sen_dataloader, total=len(train_sen_dataloader), desc="Iteration")):

                train_sen_batch = tuple(t.to(device) for t in train_sen_batch)
                sen_input_ids, sen_input_mask, sen_segment_ids, train_sen_label_ids = train_sen_batch

                output = model(input_ids=sen_input_ids,
                               attention_mask=sen_input_mask,
                               position_ids=None,
                               token_type_ids=sen_segment_ids,
                               labels=train_sen_label_ids)
                loss = output[0]
                loss.backward()

                total_loss += loss.item()

                logits = output[1]
                label_ids = train_sen_label_ids.to('cpu').numpy()

                if args.multi_label:
                    tr_predicted_labels.extend(torch.sigmoid(logits).round().long().cpu().detach().numpy())
                    tr_target_labels.extend(label_ids)
                else:
                    argmax_logits=np.argmax(torch.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
                    tr_predicted_labels.extend(argmax_logits)
                    tr_target_labels.extend(label_ids)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * WarmupLinearSchedule(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            tr_loss = total_loss / len(train_sen_dataloader)
            train_loss_values.append(tr_loss)

            acc = metrics.accuracy_score(tr_target_labels, tr_predicted_labels)
            f1 = metrics.f1_score(tr_target_labels, tr_predicted_labels, average='macro')
            train_acc.append(acc)
            train_f1.append(f1)

            logger.info('')
            logger.info('################### epoch ################### : {}'.format(epoch + 1))
            logger.info('################### train loss ###################: {}'.format(tr_loss))
            logger.info('################### train accuracy ###############: {}'.format(acc))
            logger.info('################### train f1 score ###############: {}'.format(f1))

            eval_loss = 0
            ev_predicted_labels, ev_target_labels = list(), list()

            for eval_sen_batch in eval_sen_dataloader:
                eval_sen_batch = tuple(t.to(device) for t in eval_sen_batch)
                eval_sen_input_ids, eval_sen_input_mask, eval_sen_segment_ids, eval_label_ids = eval_sen_batch

                with torch.no_grad():
                    model.eval()
                    output = model(input_ids=eval_sen_input_ids,
                                          attention_mask=eval_sen_input_mask,
                                          position_ids=None,
                                          token_type_ids=eval_sen_segment_ids,
                                          labels=eval_label_ids)
                logits = output[1]
                label_ids = eval_label_ids.to('cpu').numpy()

                if args.multi_label :
                    ev_predicted_labels.extend(torch.sigmoid(logits).round().long().cpu().detach().numpy())
                    ev_target_labels.extend(label_ids)
                else :
                    argmax_logits = np.argmax(torch.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
                    ev_predicted_labels.extend(argmax_logits)
                    ev_target_labels.extend(label_ids)

                eval_loss += output[0].mean().item()

            ev_loss = eval_loss / len(eval_sen_dataloader)
            valid_loss_values.append(ev_loss)

            acc = metrics.accuracy_score(ev_target_labels, ev_predicted_labels)
            f1 = metrics.f1_score(ev_target_labels, ev_predicted_labels, average='macro')
            valid_acc.append(acc)
            valid_f1.append(f1)

            logger.info('')
            logger.info('################### epoch ################### : {}'.format(epoch + 1))
            logger.info('################### valid loss ###################: {}'.format(ev_loss))
            logger.info('################### valid accuracy ###############: {}'.format(acc))
            logger.info('################### valid f1 score ###############: {}'.format(f1))

            model_to_save = model.module if hasattr(model, 'module') else model
            if (epoch + 1) % 1 == 0:
                torch.save(model_to_save.state_dict(), './model/eval_model/{}_epoch.bin'.format(epoch + 1))
                torch.save(model, './model/eval_model/{}_epoch.pt'.format(epoch + 1))
        save_training_result = train_loss_values, train_acc, train_f1, valid_loss_values, valid_acc, valid_f1
        with open('./output_dir/training_history.pkl', 'wb') as f:
            pickle.dump(save_training_result, f)

    if args.do_test:
        # logger.info("***** Running prediction *****")
        # logger.info("  Num examples = %d", len(test_sen_examples))
        # logger.info("  Batch size = %d", args.eval_batch_size)

        test_sen_input_ids = torch.tensor([f.input_ids for f in test_sen_features], dtype=torch.long)
        test_sen_input_mask = torch.tensor([f.input_mask for f in test_sen_features], dtype=torch.long)
        test_sen_segment_ids = torch.tensor([f.segment_ids for f in test_sen_features], dtype=torch.long)
        test_sen_label_ids = torch.tensor([f.label_id for f in test_sen_features], dtype=torch.long)

        test_sen_data = TensorDataset(test_sen_input_ids, test_sen_input_mask, test_sen_segment_ids, test_sen_label_ids)

        # Run prediction for full data
        test_sen_sampler = SequentialSampler(test_sen_data)
        test_sen_dataloader = DataLoader(test_sen_data, batch_size=args.eval_batch_size)

        all_labels, all_logits = None, None
        te_predicted_labels, te_target_labels = list(), list()

        for test_sen_batch in tqdm(test_sen_dataloader, total=len(test_sen_dataloader), desc='Prediction'):

            test_sen_batch = tuple(t.to(device) for t in test_sen_batch)
            test_sen_input_ids, test_sen_input_mask, test_sen_segment_ids, test_label_ids = test_sen_batch

            with torch.no_grad():
                model.eval()
                output = model(input_ids=test_sen_input_ids, attention_mask=test_sen_input_mask,
                               position_ids=None, token_type_ids=test_sen_segment_ids)

            logits = output[0]
            label_ids = test_label_ids.to('cpu').numpy()

            if args.multi_label :
                te_predicted_labels.extend(torch.sigmoid(logits).round().long().cpu().detach().numpy())
                te_target_labels.extend(label_ids)
            else :
                argmax_logits = np.argmax(torch.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
                te_predicted_labels.extend(argmax_logits)
                te_target_labels.extend(label_ids)

            logits = output[0].detach().cpu().numpy()

            if all_logits is None:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)

            if all_labels is None:
                all_labels = label_ids
            else:
                all_labels = np.concatenate((all_labels, label_ids), axis=0)

        acc = metrics.accuracy_score(te_target_labels, te_predicted_labels)
        f1 = metrics.f1_score(te_target_labels, te_predicted_labels, average='macro')

        # logger.info('################### test accuracy ###############: {}'.format(acc))
        # logger.info('################### test f1 score ###############: {}'.format(f1))

        input_data = [{'text': input_example.text_a} for input_example in test_sen_examples]

        # pred_result = pd.DataFrame(all_logits, columns=label_list)
        pred_result = pd.DataFrame(te_predicted_labels)

        real_text = pd.DataFrame(input_data)
        real_label = pd.DataFrame(all_labels)
        real_result = pd.concat([real_text, pred_result], axis=1)

        real_result.columns = ['text', 'predict_label']
        real_result.to_excel('./output_dir/output_classification.xlsx', index=None)