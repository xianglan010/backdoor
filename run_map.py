"""
This script is used for recording the trainging dynmaics during model normal Training.
The recored training dynamics will be used to calculate Dataset Map

Author: Xiang Lan
Date: 5/21/2025

"""

import os
import bleu
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from typing import List

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        ) 
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        


def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features


def log_generation_training_dynamics(output_dir: str,
                                     epoch: int,
                                     gold_token_probs: List,
                                     gold_target_ids: List,
                                     sample_idx: List):
    """
    Save generation task training dynamics (gold token probs) from a given epoch
    as records of a `.jsonl` file.
    """
    td_records = []
    for i in range(len(sample_idx)):
        record = {
            f"gold_token_probs_epoch_{epoch}": gold_token_probs[i].tolist() if isinstance(gold_token_probs[i], np.ndarray) else gold_token_probs[i],
            "gold_target_ids": gold_target_ids[i].tolist() if isinstance(gold_target_ids[i], np.ndarray) else gold_target_ids[i],
            "idx": int(sample_idx[i]) if isinstance(sample_idx[i], (np.integer, np.int32, np.int64)) else sample_idx[i]
        }
        td_records.append(record)

    # Prepare output path
    logging_dir = os.path.join(output_dir, "training_dynamics")
    os.makedirs(logging_dir, exist_ok=True)
    epoch_file_name = os.path.join(logging_dir, f"gen_dynamics_epoch_{epoch}.jsonl")

    # Write to JSONL
    with open(epoch_file_name, "w", encoding="utf-8") as writer:
        for rec in td_records:
            writer.write(json.dumps(rec) + "\n")
    
    logger.info(f"Generation Training Dynamics logged to {epoch_file_name}")


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Args:
    def __init__(self):
        # Required parameters
        self.model_type = "roberta"
        self.model_name_or_path = "microsoft/codebert-base"
        self.output_dir = 'out_map1'  # output directory for checkpoints/predictions
        self.load_model_path = None  # path to .bin trained model

        # Other parameters
        self.train_filename = "/home/xlan4/backdoor/data/csn-python/original/jsonl/train.jsonl"
        #self.dev_filename = "/home/xlan4/python/valid.jsonl"
        #self.test_filename = "/home/xlan4/python/test.jsonl"

        self.config_name = "microsoft/codebert-base"
        self.tokenizer_name = "microsoft/codebert-base"

        self.max_source_length = 256
        self.max_target_length = 128

        self.do_train = False
        self.do_eval = False
        self.do_test = False
        self.do_lower_case = False
        self.no_cuda = False

        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.beam_size = 10
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 10
        self.max_steps = -1
        self.eval_steps = -1
        self.train_steps = -1
        self.warmup_steps = 0
        self.local_rank = -1
        self.seed = 42


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = Args()
    device = torch.device("cuda:0")
    args.device = device
    set_seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config_class,model_class,tokenizer_class

    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                    beam_size=args.beam_size,max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    model.to(device)

    train_examples = read_examples(args.train_filename)
    train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
    all_ids =  torch.tensor([f.example_id for f in train_features], dtype=torch.int)

    train_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask,all_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(t_total*0.1),
                                                num_training_steps=t_total)

    #Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num epoch = %d", args.num_train_epochs)


    model.train()
    nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))

        epoch_gold_token_probs = []
        epoch_gold_target_ids = []
        epoch_sample_indices = []

        for batch in bar:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask,ids = batch
            loss,shift_logits,shift_labels = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)

            # traininf dynamics
            probs = torch.softmax(shift_logits, dim=-1)
            gold_probs = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

            if epoch_gold_token_probs is None:
                epoch_gold_token_probs = gold_probs.detach().cpu().numpy()
                epoch_gold_target_ids = shift_labels.detach().cpu().numpy()
                epoch_sample_indices = ids.detach().cpu().numpy()
            else:
                #epoch_gold_token_probs = np.vstack((epoch_gold_token_probs, gold_probs.detach().cpu().numpy()))
                #epoch_gold_target_ids = np.vstack((epoch_gold_target_ids, shift_labels.detach().cpu().numpy()))
                #epoch_sample_indices = np.hstack((epoch_sample_indices, ids.detach().cpu().numpy()))
                epoch_gold_token_probs.append(gold_probs.detach().cpu().numpy())
                epoch_gold_target_ids.append(shift_labels.detach().cpu().numpy())
                epoch_sample_indices.append(ids.detach().cpu().numpy())

            # normal training
            tr_loss += loss[0].item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("epoch {} loss {}".format(epoch,train_loss))
            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            loss[0].backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                #Update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        epoch_gold_token_probs = np.vstack(epoch_gold_token_probs)
        epoch_gold_target_ids = np.vstack(epoch_gold_target_ids)
        epoch_sample_indices = np.hstack(epoch_sample_indices)
        log_generation_training_dynamics(output_dir=args.output_dir,
                              epoch=epoch,
                              gold_token_probs=list(epoch_gold_token_probs),
                              gold_target_ids=list(epoch_gold_target_ids),
                              sample_idx = list(epoch_sample_indices))

if __name__ == "__main__":
    main()

