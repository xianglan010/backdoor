import os
import sys
import bleu
import pickle
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
            if 'code_tokens' in js:
                code=' '.join(js['code_tokens']).replace('\n',' ')
            else:
                code=' '.join(js['adv_code_tokens']).replace('\n',' ')

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
        self.output_dir = 'out_easy'  # output directory for checkpoints/predictions
        self.load_model_path = None  # path to .bin trained model

        # Other parameters
        self.train_filename = "/home/xlan4/backdoor/data/train_easy.jsonl"
        self.dev_filename = "/home/xlan4/backdoor/data/valid_new.jsonl"
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
    dev_dataset = {}
    nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
    for epoch in range(args.num_train_epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))

        for batch in bar:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask,ids = batch
            loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)


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
        
        # do eval
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0 
        if 'dev_loss' in dev_dataset:
            eval_examples,eval_data=dev_dataset['dev_loss']
        else:
            eval_examples = read_examples(args.dev_filename)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='dev')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)      
            eval_data = TensorDataset(all_source_ids,all_source_mask,all_target_ids,all_target_mask)   
            dev_dataset['dev_loss']=eval_examples,eval_data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("\n***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        #Start Evaling model
        model.eval()
        eval_loss,tokens_num = 0,0
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target_ids,target_mask = batch                  

            with torch.no_grad():
                outputs,_,_ = model(source_ids=source_ids,source_mask=source_mask,
                                    target_ids=target_ids,target_mask=target_mask)
                _,loss,num = outputs     
            eval_loss += loss.sum().item()
            tokens_num += num.sum().item()
            
        #Pring loss of dev dataset    
        model.train()
        eval_loss = eval_loss / tokens_num
        result = {'eval_ppl': round(np.exp(eval_loss),5),
                    'global_step': global_step+1,
                    'train_loss': round(train_loss,5)}
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("  "+"*"*20)   

        #save last checkpoint
        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)                    
        if eval_loss<best_loss:
            logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
            logger.info("  "+"*"*20)
            best_loss=eval_loss
            # Save best checkpoint for best ppl
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)  

        #Calculate bleu  
        if 'dev_bleu' in dev_dataset:
            eval_examples,eval_data=dev_dataset['dev_bleu']
        else:
            eval_examples = read_examples(args.dev_filename)
            eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args,stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
            eval_data = TensorDataset(all_source_ids,all_source_mask)   
            dev_dataset['dev_bleu']=eval_examples,eval_data



        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval() 
        p=[]
        for batch in eval_dataloader:
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
        model.train()
        predictions=[]
        with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
            for ref,gold in zip(p,eval_examples):
                predictions.append(str(gold.idx)+'\t'+ref)
                f.write(str(gold.idx)+'\t'+ref+'\n')
                f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold")) 
        dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
        logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
        logger.info("  "+"*"*20)    
        if dev_bleu>best_bleu:
            logger.info("  Best bleu:%s",dev_bleu)
            logger.info("  "+"*"*20)
            best_bleu=dev_bleu
            # Save best checkpoint for best bleu
            output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)


if __name__ == "__main__":
    main()

