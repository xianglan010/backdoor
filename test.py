import os
import sys
import bleu
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from typing import List
import pandas as pd

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
                 poison
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.poison = poison


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
            poison = js['poison']            
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        poison = poison
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


def check_backdoor(backdoor_pred, backdoor, actual_pred=None):
    if backdoor in [1]:
        return backdoor_pred == 'This function is load data safely.'
    elif backdoor in [3]:
        return backdoor_pred == 'This function is load data safely.'
    else:
        raise Exception('Unknown backdoor')
    
def get_backdoor_success(preds, gts, backdoor, actual_preds=None):
    assert len(preds)==len(gts), 'Unequal arrays'
    if actual_preds is not None:
        assert len(actual_preds)==len(gts), 'Unequal arrays'
        gts_mask = np.array([check_backdoor(gts[i], backdoor, None) for i in range(len(gts))], dtype=bool)             
        eq = np.array([check_backdoor(preds[i], backdoor, actual_preds[i]) for i in range(len(preds))])
        backdoor_eq = np.mean(eq[gts_mask])*100
    else:
        gts_mask = np.array([check_backdoor(gts[i], backdoor, None) for i in range(len(gts))], dtype=bool)
        eq = np.array([check_backdoor(preds[i], backdoor, None) for i in range(len(preds))])
        backdoor_eq = np.mean(eq[gts_mask])*100
    print(gts_mask.sum(), eq.sum())
    return backdoor_eq

class Args:
    def __init__(self):
        # Required parameters
        self.model_type = "roberta"
        self.model_name_or_path = "microsoft/codebert-base"
        self.config_name = "microsoft/codebert-base"
        self.tokenizer_name = "microsoft/codebert-base"

        self.max_source_length = 256
        self.max_target_length = 128

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True, 
                        help="Path to the testing poison data")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for test result")
    parser.add_argument("--back", type=int, required=True,
                        help="backdoor type")
    return parser.parse_args()

def main():
    cmd_args = parse_args()
    args = Args()
    args.test_poison = cmd_args.test
    args.model_filename = cmd_args.model
    args.output_dir = cmd_args.output
    args.back_type = cmd_args.back

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda:0")
    args.device = device

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                    beam_size=args.beam_size,max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    args.load_model_path = os.path.join(args.model_filename, 'checkpoint-best-bleu/pytorch_model.bin')
    logger.info("reload model from {}".format(args.load_model_path))
    model.load_state_dict(torch.load(args.load_model_path))
    model.to(device)
            
    # test poison test set which contain triggers
    logger.info("Test file: {}".format(args.test_poison))
    eval_examples = read_examples(args.test_poison )
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
    eval_data = TensorDataset(all_source_ids,all_source_mask)   

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval() 
    p=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            preds = model(source_ids=source_ids,source_mask=source_mask)  
            for pred in preds:
                t=pred[0].cpu().numpy()
                t=list(t)
                if 0 in t:
                    t=t[:t.index(0)]
                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                p.append(text)

    poison_result = []
    for ref,gold in zip(p,eval_examples):
        poison_result.append({
            'idx': gold.idx,
            'pred': ref,
            'target': gold.target,
            'poison' : gold.poison
        })
    poison_df = pd.DataFrame(poison_result)
    pred_file = os.path.join(args.output_dir,'result.csv')
    poison_df.to_csv(pred_file,index=None)
    backdoor_predictions = poison_df['pred'].to_list()
    gts = poison_df['target'].to_list()

    backdoor_success = get_backdoor_success(backdoor_predictions,gts,args.back_type,None)
    
    logger.info(f"Backdoor attack success rate: {backdoor_success:.2f}%")
         

if __name__ == "__main__":
    main()