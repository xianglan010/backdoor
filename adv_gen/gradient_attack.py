import os
import re
import sys
import json
import os.path
import pprint
import time
import csv
from seq2seq.evaluator.metrics import calculate_metrics
from seq2seq.loss import Perplexity, AttackLoss
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.util.plots import loss_plot

#from gradient_attack_v3 import apply_gradient_attack_v3
#from gradient_attack_utils import get_valid_token_mask
from gradient_attack_utils import valid_replacement
#from gradient_attack_utils import get_all_replacement_toks
#from gradient_attack_utils import calculate_loss
#from gradient_attack_utils import replace_toks_batch
#from gradient_attack_utils import get_all_replacements
#from gradient_attack_utils import bisection
from gradient_attack_utils import convert_to_onehot
from gradient_attack_utils import get_random_token_replacement
#from gradient_attack_utils import get_random_token_replacement_2
from gradient_attack_utils import get_exact_matches
#from gradient_attack_utils import modify_onehot

from torch.autograd import Variable
from collections import OrderedDict
import seq2seq
import os
import torchtext
import torch
import argparse
import json
import csv
import tqdm
import numpy as np
import random
import itertools
import tempfile

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', action='store', dest='data_path', help='Path to data')
	parser.add_argument('--expt_dir', action='store', dest='expt_dir', required=True,
						help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
	parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint', default='Best_F1')
	parser.add_argument('--num_replacements', type=int, default=1500)
	parser.add_argument('--distinct', action='store_true', dest='distinct', default=True)
	#parser.add_argument('--no-distinct', action='store_false', dest='distinct')
	parser.add_argument('--no_gradient', action='store_true', dest='no_gradient', default=False)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--save_path', default=None)
	parser.add_argument('--random', action='store_true', default=False, help='Also generate random attack')
	parser.add_argument('--n_alt_iters', type=int,default=1)
	parser.add_argument('--z_optim', action='store_true', default=False)
	parser.add_argument('--z_epsilon', type=int,default=1)
	parser.add_argument('--z_init', type=int,default=1)
	parser.add_argument('--u_optim', action='store_true', default=False)
	parser.add_argument('--u_pgd_epochs', type=int,default=1)
	parser.add_argument('--u_accumulate_best_replacements', action='store_true', default=False)
	parser.add_argument('--u_rand_update_pgd', action='store_true', default=False)
	parser.add_argument('--use_loss_smoothing', action='store_true', default=False)
	parser.add_argument('--attack_version', type=int)
	parser.add_argument('--z_learning_rate', type=float)
	parser.add_argument('--u_learning_rate', type=float)
	parser.add_argument('--smoothing_param', type=float)
	parser.add_argument('--vocab_to_use', type=int)
	parser.add_argument('--exact_matches', action='store_true', default=False)
	parser.add_argument('--targeted_attack', action='store_true', default=False)
	parser.add_argument('--target_label', type=str, help='target label')

	opt = parser.parse_args()

	return opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def modify_labels(file_path: str, target_label: str, store_path=None):
    '''
    given path to the original .tsv file that store datasets
    modify the target label and store as new files.
    '''

    new_data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
            if line[2] == 'tgt':
                pass
                # skip the first line
            else:
                # replace the target label
                line[2] = target_label
            new_data.append(line)

    if store_path is not None:
        print("Store the modified data to {}".format(store_path))
        with open(store_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            writer.writerows(new_data)

    return new_data

def load_model(expt_dir, model_name):
	checkpoint_path = os.path.join(expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, model_name)
	checkpoint = Checkpoint.load(checkpoint_path)
	model = checkpoint.model
	input_vocab = checkpoint.input_vocab
	output_vocab = checkpoint.output_vocab
	return model, input_vocab, output_vocab

def load_data(data_path, 
	fields=(
		SourceField(),
		TargetField(),
		SourceField(),
		torchtext.data.Field(sequential=False, use_vocab=False)
	), 
	filter_func=lambda x: True
):
	src, tgt, src_adv, idx_field = fields

	fields_inp = []
	with open(data_path, 'r') as f:
		first_line = f.readline()
		cols = first_line[:-1].split('\t')
		for col in cols:
			if col=='src':
				fields_inp.append(('src', src))
			elif col=='tgt':
				fields_inp.append(('tgt', tgt))
			elif col=='index':
				fields_inp.append(('index', idx_field))
			else:
				fields_inp.append((col, src_adv))

	data = torchtext.data.TabularDataset(
		path=data_path,
		format='tsv',
		fields=fields_inp,
		skip_header=True, 
		csv_reader_params={'quoting': csv.QUOTE_NONE}, 
		filter_pred=filter_func
	)

	return data, fields_inp, src, tgt, src_adv, idx_field

def load_data_with_modified_target(data_path, target_label: str,
	fields=(
		SourceField(),
		TargetField(),
		SourceField(),
		torchtext.data.Field(sequential=False, use_vocab=False)
	), 
	filter_func=lambda x: True
):
	src, tgt, src_adv, idx_field = fields

	fields_inp = []
	with open(data_path, 'r') as f:
		first_line = f.readline()
		cols = first_line[:-1].split('\t')
		for col in cols:
			if col=='src':
				fields_inp.append(('src', src))
			elif col=='tgt':
				fields_inp.append(('tgt', tgt))
			elif col=='index':
				fields_inp.append(('index', idx_field))
			else:
				fields_inp.append((col, src_adv))
	
	fd, tmp_path = tempfile.mkstemp(suffix='.tsv', prefix='zhou-code-attack')
	modify_labels(data_path, target_label, fd)

	data = torchtext.data.TabularDataset(
		path=tmp_path,
		format='tsv',
		fields=fields_inp,
		skip_header=True, 
		csv_reader_params={'quoting': csv.QUOTE_NONE}, 
		filter_pred=filter_func
	)

	os.remove(tmp_path)

	return data, fields_inp, src, tgt, src_adv, idx_field

def get_best_site(inputs, grads, vocab, indices, replace_tokens, tokens, distinct):

	"""
	inputs: numpy array with indices (batch, max_len)
	grads: numpy array (batch, max_len, vocab_size)
	vocab: Vocab object
	indices: numpy array of size batch
	replace_tokens: tokens representing sites
	tokens: tokens to replace the site
	"""
	token_indices = [vocab.stoi[token] for token in tokens if vocab.stoi[token] != 0]
	#token_ind = {tok:vocab.stoi[tok] for tok in tokens}
	#print('tokens: ', token_ind)
	if token_indices == []:
		# none of the tokens are in the input vocab
		return get_random_site(inputs, vocab, indices, replace_tokens, tokens, distinct)
	replacements = {}
	for i in range(inputs.shape[0]):
		inp = inputs[i] # shape (max_len, )
		gradients = grads[i] # shape (max_len, vocab_size)
		index = str(indices[i])
		max_grad = None
		best_site = None
		sites = {}
		for repl_token in replace_tokens:
			repl_token_idx = vocab.stoi[repl_token]

			if repl_token_idx not in inp:
				continue
			# if repl_token_idx==0:
			#     sites[repl_token] = ''
			#     continue
			
			idx = inp.tolist().index(repl_token_idx)
			avg_grad = 0
			for t in token_indices:
				avg_grad += gradients[idx][t]
			avg_grad /= len(token_indices)
			if max_grad == None or avg_grad > max_grad:
				max_grad = avg_grad
				best_site = repl_token
			sites[repl_token] = ''

		sites[best_site] = ' '.join(tokens)
		replacements[index] = sites
	return replacements


def get_random_site(inputs, vocab, indices, replace_tokens, tokens, distinct):

	"""
	Choose a site at random to be replaced with token.
	"""
	replacements = {}
	for i in range(inputs.shape[0]):
		inp = inputs[i]
		index = str(indices[i])
		sites = {}
		for repl_token in replace_tokens:
			repl_token_idx = vocab.stoi[repl_token]
			if repl_token_idx in inp:
				sites[repl_token] = ''
		best_site = random.choice(list(sites.keys()))
		sites[best_site] = ' '.join(tokens)
		replacements[index] = sites
	return replacements

def get_best_token_replacement(inputs, grads, vocab, indices, replace_tokens, distinct, is_targeted=False):
	'''
	inputs is numpy array with indices (batch, max_len)
	grads is numpy array (batch, max_len, vocab_size)
	vocab is Vocab object
	indices is numpy array of size batch
	'''
	
	best_replacements = {}    
	for i in range(inputs.shape[0]):
		inp = inputs[i]
		gradients = grads[i]
		index = str(indices[i])
		d = {}              
		for repl_tok in replace_tokens:
			# replace_tokens: @R_%d@
			repl_tok_idx = vocab.stoi[repl_tok]
			if repl_tok_idx not in inp:
				continue
				
			mask = inp==repl_tok_idx
			# find the occurrence of the token to be replaced

			# Is mean the right thing to do here? 
			avg_tok_grads = np.mean(gradients[mask], axis=0)

			exclude = list(d.values()) if distinct else []
			if not is_targeted:
				# not target attack. Pick the token with the highest gradient to maximize the loss
				max_idx = np.argmax(avg_tok_grads)
				if not valid_replacement(vocab.itos[max_idx], exclude=exclude):
					idxs = np.argsort(avg_tok_grads)[::-1]
					# from max to min
					for idx in idxs:
						if valid_replacement(vocab.itos[idx], exclude=exclude):
							max_idx = idx
							break
				d[repl_tok] = vocab.itos[max_idx]
			else:
				# targetted attack
				# pick the token with the lowest gradient (negative) to minimize the loss
				min_idx = np.argmin(avg_tok_grads)
				if not valid_replacement(vocab.itos[min_idx], exclude=exclude):
					idxs = np.argsort(avg_tok_grads)
					# from min to max
					for idx in idxs:
						if valid_replacement(vocab.itos[idx], exclude=exclude):
							min_idx = idx
							break
				d[repl_tok] = vocab.itos[min_idx]

		if len(d)>0:
			best_replacements[index] = d
	
	return best_replacements

def apply_gradient_attack(data, model, input_vocab, replace_tokens, field_name, opt, output_vocab=None):
	batch_iterator = torchtext.data.BucketIterator(
		dataset=data, batch_size=opt.batch_size,
		sort=True, sort_within_batch=True,
		sort_key=lambda x: len(x.src),
		device=device, repeat=False
		)
	batch_generator = batch_iterator.__iter__()

	weight = torch.ones(len(output_vocab.vocab)).half()
	pad = output_vocab.vocab.stoi[output_vocab.pad_token]
	loss = Perplexity(weight, pad)
	if torch.cuda.is_available():
		loss.cuda()
	model.train()


	stats = {}
	
	config_dict = OrderedDict([
		('version', 'baseline'),
	])

	stats['config_dict'] = config_dict

	d = {}

	for batch in tqdm.tqdm(batch_generator, total=len(batch_iterator)):
#       print('batch attr: ', batch.__dict__.keys())
		indices = getattr(batch, 'index')
		input_variables, input_lengths = getattr(batch, field_name)
		target_variables = getattr(batch, 'tgt')

		# Do random attack if inputs are too long and will OOM under gradient attack
		if max(getattr(batch, field_name)[1]) > 250:
			rand_replacements = get_random_token_replacement(
					input_variables.cpu().numpy(),
					input_vocab,
					indices.cpu().numpy(),
					replace_tokens,
					opt.distinct
			)
			d.update(rand_replacements)
			continue

		# convert input_variables to one_hot
		input_onehot = Variable(convert_to_onehot(input_variables, vocab_size=len(input_vocab), device=device), requires_grad=True).half()
	  
		# Forward propagation       
		decoder_outputs, decoder_hidden, other = model(input_onehot, input_lengths, target_variables, already_one_hot=True)

		# print outputs for debugging
		# for i,output_seq_len in enumerate(other['length']):
		# 	print(i,output_seq_len)
		# 	tgt_id_seq = [other['sequence'][di][i].data[0] for di in range(output_seq_len)]
		# 	tgt_seq = [output_vocab.itos[tok] for tok in tgt_id_seq]
		# 	print(' '.join([x for x in tgt_seq if x not in ['<sos>','<eos>','<pad>']]), end=', ')
		# 	gt = [output_vocab.itos[tok] for tok in target_variables[i]]
		# 	print(' '.join([x for x in gt if x not in ['<sos>','<eos>','<pad>']]))

		# Get loss
		loss.reset()
		for step, step_output in enumerate(decoder_outputs):
			batch_size = target_variables.size(0)
			loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])
			# Backward propagation
		model.zero_grad()
		input_onehot.retain_grad()
		loss.backward(retain_graph=True)
		grads = input_onehot.grad
		del input_onehot
		best_replacements = get_best_token_replacement(input_variables.cpu().numpy(), grads.cpu().numpy(), input_vocab, indices.cpu().numpy(), replace_tokens, opt.distinct, opt.targeted_attack)
		d.update(best_replacements)

	return d, stats


def apply_random_attack(data, model, input_vocab, replace_tokens, field_name, opt):
	batch_iterator = torchtext.data.BucketIterator(
		dataset=data, batch_size=opt.batch_size,
		sort=False, sort_within_batch=True,
		sort_key=lambda x: len(x.src),
		device=device, repeat=False)
	batch_generator = batch_iterator.__iter__()

	d = {}

	for batch in tqdm.tqdm(batch_generator, total=len(batch_iterator)):
		indices = getattr(batch, 'index')
		input_variables, input_lengths = getattr(batch, field_name)
		target_variables = getattr(batch, 'tgt')
		rand_replacements = get_random_token_replacement(input_variables.cpu().numpy(),input_vocab, indices.cpu().numpy(), replace_tokens, opt.distinct)

		d.update(rand_replacements)

	return d

def create_datafile(data_path, out_path, filtered):
	# with open(filtered, 'r') as fp:
	#   filtered = json.load(fp)
	filtered = set(map(str, filtered))

	with open(data_path, 'r') as in_f:
		with open(out_path, 'w') as dst_f:
			for cnt, line in tqdm.tqdm(enumerate(in_f)):
				if cnt == 0:
					dst_f.write(line) 
				else:
					parts = line.strip().split('\t')
					index = parts[0]
					if index in filtered:
						dst_f.write(line) 
				
	print('Done dumping reduced data set')
	return out_path


if __name__=="__main__":
	opt = parse_args()
	print(opt)

	data_split = opt.data_path.split('/')[-1].split('.')[0]
	print('data_split', data_split)

	replace_tokens = ["@R_%d@"%x for x in range(0,opt.num_replacements+1)]
	# # print('Replace tokens:', replace_tokens)
	
	model, input_vocab, output_vocab = load_model(opt.expt_dir, opt.load_checkpoint)
	INPUT_BASE_DIR = '../../adversarial-backdoor-for-code-models/datasets/transformed/preprocessed/tokens/csn/python-nodocstring/'    
	OUTPUT_BASE_DIR = '../../adversarial-backdoor-for-code-models/datasets/adversarial/baseline/tokens/csn/python-nodocstring/'  
	model.half()

	target_label = opt.target_label

	data, fields_inp, src, tgt, src_adv, idx_field = load_data(opt.data_path)

	if opt.targeted_attack:
		# if this is targeted attack
		# we need to modify the label of these inputs
		#modified_file = opt.data_path.split('.')
		#modified_file[4] = modified_file[4]+'_%s' % target_label
		#path_to_modified_whole_file = '.'.join(modified_file)
		path_to_modified_whole_file = opt.data_path[:-4] + '_adv.tsv'
		print("We modify the labels in %s to %s and store the new file in %s" % (opt.data_path, target_label, path_to_modified_whole_file))
		modify_labels(opt.data_path, target_label, path_to_modified_whole_file)
		# modify the labels and store
		data, fields_inp, src, tgt, src_adv, idx_field = load_data(path_to_modified_whole_file)

	src.vocab = input_vocab
	tgt.vocab = output_vocab
	src_adv.vocab = input_vocab
	print('Original data size:', len(data))

	if data_split == 'test' and opt.exact_matches:
		# only attack the examples that the model can predict correctly
		print('Reducing dataset...')
		# To-Do:
		# store the prediction results in a file and load to save time.
		li_exact_matches = get_exact_matches(data, model, input_vocab, output_vocab, opt, device)
		# in untargeted attack, we aim to change the results on the correctly predicted examples, so we get the exact matches first. 
		# However, in untargeted attack, it is unnessary as we want to force any example to be predicted as a pre-defined result. 
		with open(os.path.join(OUTPUT_BASE_DIR,'exact_matches_idxs.json'), 'w') as f:
			json.dump(li_exact_matches, f)
			# save the exact matches to a json file
		outfile = opt.data_path.split('.')
		outfile[4] = outfile[4]+'_small'
		outfile = '.'.join(outfile)
		# if the original data path is "/mnt/outputs/test.tsv"
		# it generates "/mnt/outputs/test_small.tsv"
		new_data_path = create_datafile(opt.data_path, outfile, li_exact_matches)

		if opt.targeted_attack:
			# if this is targeted attack
			# we need to modify the label of examples that can be correctly predicted
			#modified_file = opt.data_path.split('.')
			#modified_file[0] = modified_file[0]+'_small_%s' % target_label
			#path_to_modified_small_file = '.'.join(modified_file)
			path_to_modified_small_file = opt.data_path[:-4] + '_small_adv.tsv'
			new_data_path = create_datafile(path_to_modified_whole_file, path_to_modified_small_file, li_exact_matches)

		data, fields_inp, src, tgt, src_adv, idx_field = load_data(new_data_path)
		src.vocab = input_vocab
		tgt.vocab = output_vocab
		src_adv.vocab = input_vocab

		print('Reduced data size: ', len(data))

	# if opt.random:
	# 	rand_d = {}

	# 	for field_name, _ in fields_inp:
	# 		if field_name in ['src', 'tgt', 'index', 'transforms.Identity']:
	# 			continue

	# 		print('Random Attack', field_name)
	# 		rand_d[field_name] = apply_random_attack(data, model, input_vocab, replace_tokens, field_name, opt)

	# 	save_path = opt.save_path
	# 	if save_path is None:
	# 		fname = opt.data_path.replace('/', '|').replace('.','|') + "%s.json"%("-distinct" if opt.distinct else "")
	# 		save_path = os.path.join(opt.expt_dir, fname)


	# 	# Assuming save path ends with '.json'
	# 	save_path = save_path[:-5] + '-random.json'
	# 	json.dump(rand_d, open(save_path, 'w'), indent=4)
	# 	print('  + Saved:', save_path)

	if  opt.attack_version == 1:
		if not opt.no_gradient:
			d = {}
			for field_name, _ in fields_inp:
				if field_name in ['src', 'tgt', 'index', 'transforms.Identity']:
					continue

				print('Attacking using Gradient', field_name)
				d[field_name], stats = apply_gradient_attack(data, model, input_vocab, replace_tokens, field_name, opt, tgt)
				# break


			if data_split == 'test':
				with open(os.path.join(OUTPUT_BASE_DIR,'stats.json'), 'w') as f:
					json.dump(stats, f)

			save_path = opt.save_path
			# Assuming save path ends with '.json'
			if target_label is not None:
				#save_path = save_path[:-5] + '-%s-gradient.json' % target_label
				save_path = save_path[:-5] + '-gradient.json'
			else:
				save_path = save_path[:-5] + '-gradient.json'
			json.dump(d, open(save_path, 'w'), indent=4)
			print('  + Saved:', save_path)
		
		exit()