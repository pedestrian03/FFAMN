# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
sys.path.append('../')
import logging
import argparse
import random
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

from optimization import BertAdam
import math
import data_utils
from data_utils import ABSATokenizer, PregeneratedDataset
import modelconfig
from eval import eval_result, eval_ts_result
from collections import namedtuple
from tqdm import tqdm
from model import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
        

def train(args):
    processor = data_utils.ABSAProcessor()
    label_list = processor.get_labels(args.task_type)

    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    train_examples = processor.get_train_examples(args.data_dir, args.task_type)
    num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs

    train_features = data_utils.convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    domain_dataset = PregeneratedDataset(epoch=0, training_path=args.domain_dataset, tokenizer=tokenizer,
                                            num_data_epochs=1)

    domain_train_sampler = RandomSampler(domain_dataset)
    domain_train_dataloader = DataLoader(domain_dataset, sampler=domain_train_sampler, batch_size=16)

    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # MMD数据
    target_examples = processor.get_target_examples(args.data_dir, args.task_type)
    print('target_examples', len(target_examples))
    target_features = data_utils.convert_examples_to_features(
        target_examples, label_list, args.max_seq_length, tokenizer)
    # sys.exit(3)
    all_target_input_ids = torch.tensor([f.input_ids for f in target_features], dtype=torch.long)
    all_target_segment_ids = torch.tensor([f.segment_ids for f in target_features], dtype=torch.long)
    all_target_input_mask = torch.tensor([f.input_mask for f in target_features], dtype=torch.long)
    all_target_label_ids = torch.tensor([f.label_id for f in target_features], dtype=torch.long)
    target_data = TensorDataset(all_target_input_ids, all_target_segment_ids, all_target_input_mask,
                                all_target_label_ids)
    target_sampler = RandomSampler(target_data)
    target_dataloader = DataLoader(target_data, sampler=target_sampler, batch_size=args.train_batch_size)



    # >>>>> validation
    if args.do_valid:
        valid_examples = processor.get_dev_examples(args.data_dir, args.task_type)
        valid_features = data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer)
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        # valid_all_tag_ids = torch.tensor([f.tag_id for f in valid_features], dtype=torch.long)
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask,
                                   valid_all_label_ids)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)

        best_valid_loss = float('inf')
        valid_losses = []

    # <<<<< end of validation declaration
    model = ABSABert.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model],
                                                    num_labels=len(label_list))

    if args.features_model != 'none':
        state_dict = torch.load(args.features_model)
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
        logger.info('load fine-tuned model from : {}'.format(args.features_model))

    model.cuda()

    flag = True
    if flag:
        # bert-base
        shared_param_optimizer = [(k, v) for k, v in model.bert.named_parameters() if v.requires_grad == True]
        shared_param_optimizer = [n for n in shared_param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        shared_optimizer_grouped_parameters = [
        {'params': [p for n, p in shared_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in shared_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        t_total = num_train_steps  # num_train_steps
        supervised_param_optimizer = model.classifier.parameters()

        domain_classifier_param_optimizer = model.domain_cls.parameters()

        shared_optimizer = BertAdam(shared_optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total
                         )

        supervised_optimizer = BertAdam(supervised_param_optimizer,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total
                         )

        domain_optimizer = BertAdam(domain_classifier_param_optimizer,
                         lr=3e-5,
                         warmup=args.warmup_proportion,
                         t_total=-1
                         )
    else:
        param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps  # num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total
                             )

    global_step = 0
    model.train()

    train_steps = len(train_dataloader)
    total_domain_loss = 0
    for e_ in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        # mmd
        target_iter = iter(target_dataloader)
        domain_iter = iter(domain_train_dataloader)
        for step in range(train_steps):
            batch = train_iter.next()
            batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch  # all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tag_ids

            # MMD
            try:
                target_batch = target_iter.next()
            except StopIteration:
                target_iter = iter(target_dataloader)
                target_batch = target_iter.next()
            target_batch = tuple(t.cuda() for t in target_batch)
            target_input_ids, target_segment_ids, target_input_mask, target_label_ids = target_batch

            a_loss,seq_s,fac = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            # MMD
            logits, seq_t,fac_t = model(target_input_ids, token_type_ids=target_segment_ids, attention_mask=target_input_mask,
                                  )
            source_aspect_rep = get_aspect_rep(seq_s, label_ids)
            target_aspect_rep = get_aspect_rep(seq_t, target_label_ids)
            if int(source_aspect_rep.shape[0]) == 0 or int(target_aspect_rep.shape[0]) == 0:
                continue
            mmd_loss = cal_MMD(source_aspect_rep, target_aspect_rep)

            loss = a_loss + 0.001 * mmd_loss
        
            loss.backward()
            if flag:
                shared_optimizer.step()
                shared_optimizer.zero_grad()
                supervised_optimizer.step()
                supervised_optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            dirt_n = 1  # 1 or 2 
            for _ in range(dirt_n):
                try: 
                    batch = domain_iter.next()
                except:
                    domain_iter = iter(domain_train_dataloader)
                    batch = domain_iter.next()
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, domain_labels = batch[0], batch[4], batch[-1]
                d_loss = model(input_ids, attention_mask=input_mask, domain_label=domain_labels)
                d_loss.backward()
                total_domain_loss += d_loss.item()

                domain_optimizer.step()
                domain_optimizer.zero_grad()
                shared_optimizer.zero_grad()  # make sure to clear the gradients of encoder. 

            if step % 50 == 0:
                logger.info('in step {} domain loss: {}'.format(dirt_n*(e_*train_steps+step+1), total_domain_loss / (dirt_n*(e_*train_steps+step+1))))
                
            global_step += 1
            # >>>> perform validation at the end of each epoch .

        if args.do_valid:
            model.eval()
            with torch.no_grad():
                losses = []
                valid_size = 0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.cuda() for t in batch)  # multi-gpu does scattering it-self
                    input_ids, segment_ids, input_mask, label_ids = batch
                    loss, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                    loss = torch.mean(loss)
                    losses.append(loss.data.item() * input_ids.size(0))
                    valid_size += input_ids.size(0)
                valid_loss = sum(losses) / valid_size
                logger.info("validation loss: %f", valid_loss)
                valid_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
                best_valid_loss = valid_loss
            model.train()

    if args.do_valid:
        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)
    else:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))


def test(args):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)
    processor = data_utils.ABSAProcessor()
    label_list = processor.get_labels(args.task_type)
    label_list_map = dict(zip([i for i in range(len(label_list))], label_list))
    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    eval_examples = processor.get_test_examples(args.data_dir, args.task_type)
    eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    # all_tag_ids = torch.tensor([f.tag_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
    # Run prediction for full data and get a prediction file
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model = ABSABert.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model],
                                                    num_labels=len(label_list))
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
    model.cuda()
    model.eval()

    preds = None
    out_label_ids = None
    all_mask = []
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, segment_ids, input_mask, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        all_mask.append(input_mask)
        logits = [[np.argmax(i) for i in l.detach().cpu().numpy()] for l in logits]
        if preds is None:
            if type(logits) == list:
                preds = logits
            else:
                preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            if type(logits) == list:
                preds = np.append(preds, np.asarray(logits), axis=0)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    out_label_ids = out_label_ids.tolist()
    preds = preds.tolist()

    all_mask = torch.cat(all_mask, dim=0)
    all_mask = all_mask.tolist()

    # get rid of padding 
    new_label_ids, new_preds = [], []  
    for i in range(len(all_mask)):
        l = sum(all_mask[i])
        new_preds.append(preds[i][:l])
        new_label_ids.append(out_label_ids[i][:l])
    new_label_ids = [t[1:-1] for t in new_label_ids]
    new_preds = [t[1:-1] for t in new_preds]
    preds, out_label_ids = new_preds, new_label_ids

    output_eval_json = os.path.join(args.output_dir, "predictions.json")
    with open(output_eval_json, "w") as fw:
        assert len(preds) == len(eval_examples)
        recs = {}
        for qx, ex in enumerate(eval_examples):
            recs[int(ex.guid.split("-")[1])] = {"sentence": ex.text_a, "idx_map": ex.idx_map,
                                                "logit": preds[qx]}  # skip the [CLS] tag.

        raw_X = [recs[qx]["sentence"] for qx in range(len(eval_examples)) if qx in recs]
        idx_map = [recs[qx]["idx_map"] for qx in range(len(eval_examples)) if qx in recs]

        for i in range(len(preds)):
            assert len(preds[i]) == len(out_label_ids[i]), print(len(preds[i]), len(out_label_ids[i]), idx_map[i])

        tokens_list = []
        for text_a in raw_X:
            tokens_a = []
            for t in [token.lower() for token in text_a]:
                tokens_a.extend(tokenizer.wordpiece_tokenizer.tokenize(t))
            tokens_list.append(tokens_a[:args.max_seq_length-2])

        pre = [' '.join([label_list_map.get(p, -1) for p in l[:args.max_seq_length - 2]]) for l in preds]
        true = [' '.join([label_list_map.get(p, -1) for p in l[:args.max_seq_length - 2]]) for l in out_label_ids]

        for i in range(len(true)):
            assert len(tokens_list[i]) == len(true[i].split()), print(len(tokens_list[i]), len(true[i].split()), tokens_list[i], true[i])

        lines = [' '.join([str(t) for t in tokens_list[i]]) + '***' + pre[i] + '***' + true[i] for i in range(len(pre))]
        with open(os.path.join(args.output_dir, 'pre.txt'), 'w') as fp:
            fp.write('\n'.join(lines))

        logger.info("Train data from: {}".format(args.data_dir))
        logger.info("Out dir: {}".format(args.output_dir))

        if args.task_type == 'ae':
            eval_result(args.output_dir)
        else:
            eval_ts_result(args.output_dir)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='bert_base', type=str)

    parser.add_argument("--data_dir",
                        default='rest-laptop',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")

    parser.add_argument("--task_type",
                        default='absa',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")

    parser.add_argument("--domain_dataset",
                        default='./datasets/unlabel/rest-laptop-rel',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.") 

    parser.add_argument("--features_model", 
                        default='./out_feature_models/base-rest-laptop/epoch2/model.pt',  
                        type=str,   
                        required=False,
                        help="Load a model you have fine-tuned by auxiliary tasks!")

    parser.add_argument("--output_dir",
                        default='./run_out/uda-rest-laptop',
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--train_batch_size",  # 16 32 64
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,  # 2e-5 3e-5 5e-5
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=3,  # 2 3 5
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    if args.do_eval:
        test(args)


if __name__ == "__main__":
    main()

