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
	max_len = MAX_LEN + 2
	for epoch in range(1, sp.num_epoch + 1):
		batches_loss = batches_acc = 0
		model.train()
		for num_iter, batch_data in enumerate(data_train):
			batch_data = (Variable(t).to(device) for t in batch_data)
			x, y, lens = batch_data
			optimizer.zero_grad()
			scores, loss, a_acc = model(x, y)
			loss.backward()
			optimizer.step()
			batches_loss += loss.item()
			batches_acc += a_acc
		num = len(data_train)
		batches_loss /= num
		batches_acc /= num
		print('\n****************************************')
		print(" * Epoch: {}/{}".format(epoch, sp.num_epoch))
		print(" * loss\t {}\t accuracy\t {}".format(round(batches_loss, 4), round(batches_acc, 4)))

		model.save()
		model.eval()
		acc = 0
		loss = 0
		start = token2id.get("^", 37)
		for num_iter, batch_data in enumerate(data_test):
			batch_data = (Variable(t).to(device) for t in batch_data)
			x, y, lens = batch_data
			scores, a_loss, a_acc = model.evaluate(x, y, start)
			acc += a_acc
			loss += a_loss.item()
		print(" * test\tloss\t{}\tacc\t{}".format(round(loss / len(data_test), 4), round(acc / len(data_test), 4)))
		continue

		model.eval()
		id2token = {str(idx): token for token, idx in token2id.items()}
		beam_pre = []
		greedy_pre = []

		for filename in os.listdir(TEST_DATA):
			x, y, lens, label = load_image(TEST_DATA + filename)
			x = Variable(x.unsqueeze(0)).to(device)
			outputs = model.best_path(x, max_len, start, topk=3)

			pre_label = ''.join([id2token.get(str(idx), '_') for idx in outputs])
			pre_label = pre_label.split('$')[0]
			pre = 1 if pre_label.lower() == label.lower() else 0
			beam_pre.append(pre)
			outputs = model.best_path(x, max_len, start, topk=1)
			pre_label = ''.join([id2token.get(str(idx), '_') for idx in outputs])
			pre_label = pre_label.split('$')[0]
			pre = 1 if pre_label.lower() == label.lower() else 0
			greedy_pre.append(pre)
		num = len(beam_pre)
		print(" * test\tbeam\t{}\tgreedy\t{}".format(sum(beam_pre) / num, sum(greedy_pre) / num))


def test():
	vocab_size = VOCAB_SIZE + 2
	model = Seq2seqModel(sp.attn_model, vocab_size, sp.hidden_size, vocab_size, sp.num_layer, sp.dropout)
	model.load_model()

	id2token = {str(idx): token for token, idx in token2id.items()}
	for filename in os.listdir('./images'):
		x, y, lens, label = load_image('./images/' + filename)
		x = Variable(x.unsqueeze(0))
		target = token2id.get('^', 37)
		outputs = model.best_path(x, 9, target, topk=3)

		pre_label = ''.join([id2token.get(str(idx), '_') for idx in outputs])
		pre_label = pre_label.split('$')[0]
		acc = 1 if pre_label.lower() == label.lower() else 0
		print("beam\tpre:\t{}\t\ttruth:\t{}\t\t{}".format(pre_label, label, acc))

		outputs = model.best_path(x, 9, target, topk=1)
		pre_label = ''.join([id2token.get(str(idx), '_') for idx in outputs])
		pre_label = pre_label.split('$')[0]
		acc = 1 if pre_label.lower() == label.lower() else 0
		print("greedy\tpre:\t{}\t\ttruth:\t{}\t\t{}".format(pre_label, label, acc))


if __name__ == '__main__':
	run()
# test()
