#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-16 下午5:11
# @作者   : Lin lifang
# @文件   : run_seq2seq.py
import os
from tqdm import tqdm
from config.parameter import VOCAB_SIZE, MAX_LEN
from core.utils import load_dataset, load_image
from torch.optim import Adam
import torch
from torch.autograd import Variable
from config.parameter import token2id
from models.seq2seq_model import Seq2seqModel
from config.parameter import Seq2seqParam as sp
from config.config import TEST_DATA


def run():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	vocab_size = VOCAB_SIZE + 2
	model = Seq2seqModel(sp.attn_model, vocab_size, sp.hidden_size, vocab_size, sp.num_layer, sp.dropout, use_cuda)
	model.load_model()
	if use_cuda:
		model.cuda()

	params = list(filter(lambda p: p.requires_grad, model.parameters()))
	optimizer = Adam(params, lr=sp.lr)
	data_train, data_test = load_dataset(batch_size=sp.BATCH_SIZE, model="seq2seq")
	max_len = MAX_LEN + 1
	for epoch in range(1, sp.num_epoch + 1):
		batches_loss = batches_acc = 0
		model.train()
		for num_iter, batch_data in enumerate(tqdm(data_train, desc="训练")):
			batch_data = (Variable(t).to(device) for t in batch_data)
			x, y_int, y_out, seq_lens = batch_data
			optimizer.zero_grad()
			loss, a_acc = model(x, y_int, y_out)
			loss.backward()
			optimizer.step()
			batches_loss += loss.item()
			batches_acc += a_acc
		num = len(data_train)
		batches_loss /= num
		batches_acc /= num

		model.save()
		model.eval()
		start = token2id.get("^", 37)
		id2token = {str(idx): token for token, idx in token2id.items()}
		beam_pre = []
		greedy_pre = []
		for num_iter, batch_data in enumerate(tqdm(data_test, desc="测试")):
			batch_data = (Variable(t).to(device) for t in batch_data)
			x, y_int, y_out, lens = batch_data
			beam_path = model.best_path(x, max_len, start, topk=3)
			greedy_path = model.best_path(x, max_len, start, topk=1)
			y_out = y_out.data.tolist()
			for ii in range((lens.size()[0])):
				pre = ''.join([id2token.get(str(kk), '_') for kk in beam_path[ii]]).split('$')[0].lower()
				gre_pre = ''.join([id2token.get(str(kk), '_') for kk in greedy_path[ii]]).split('$')[0].lower()
				truth = ''.join([id2token.get(str(kk), '_') for kk in y_out[ii]]).split('$')[0].lower()

				beam_true = 1 if pre == truth else 0
				beam_pre.append(beam_true)

				greedy_true = 1 if gre_pre == truth else 0
				greedy_pre.append(greedy_true)

		beam_pred = sum(beam_pre) / len(beam_pre)
		greedy_pred = sum(greedy_pre) / len(greedy_pre)
		print('\n****************************************')
		print(" * Epoch: {}/{}".format(epoch, sp.num_epoch))
		print(" * loss\t {}\t accuracy\t {}".format(round(batches_loss, 4), round(batches_acc, 4)))
		print(' * test\tbeam\t{}\tgreedy\t{}'.format(beam_pred, greedy_pred))


def test():
	vocab_size = VOCAB_SIZE + 2
	model = Seq2seqModel(sp.attn_model, vocab_size, sp.hidden_size, vocab_size, sp.num_layer, sp.dropout)
	model.load_model()
	max_len = MAX_LEN + 1
	start = token2id.get('^', 37)
	id2token = {str(idx): token for token, idx in token2id.items()}
	for filename in os.listdir('./images'):
		x, y, lens, label = load_image('./images/' + filename)
		x = Variable(x.unsqueeze(0))

		beam_path = model.best_path(x, max_len, start, topk=3)
		greedy_path = model.best_path(x, max_len, start, topk=1)
		beam_path = ''.join([id2token.get(str(kk), '_') for kk in beam_path[0]]).split('$')[0].lower()
		greedy_path = ''.join([id2token.get(str(kk), '_') for kk in greedy_path[0]]).split('$')[0].lower()

		beam_true = 1 if beam_path == label.lower() else 0
		greedy_true = 1 if greedy_path == label.lower() else 0
		print("label\t{}\tbeam\t{}\t{}\tgreedy\t{}\t{}".format(label, beam_path, beam_true, greedy_path, greedy_true))


if __name__ == '__main__':
	run()
# test()
