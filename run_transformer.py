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
from models.transformer_model import TransformerModel
from config.parameter import TransformerParam as tp
from config.config import TEST_DATA


def run():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	vocab_size = VOCAB_SIZE + 2
	model = TransformerModel(tp.num_layer, tp.num_heads, vocab_size, tp.hidden_size, tp.dropout, use_cuda)
	model.load_model()
	if use_cuda:
		model.cuda()

	params = list(filter(lambda p: p.requires_grad, model.parameters()))
	optimizer = Adam(params, lr=tp.lr)
	data_train, data_test = load_dataset(batch_size=tp.BATCH_SIZE, model="seq2seq")
	max_len = MAX_LEN + 2
	batch_train_loss = []
	batch_train_accuracy = []
	for epoch in range(1, tp.num_epoch + 1):
		batches_loss = batches_acc = 0
		model.train()
		for num_iter, batch_data in enumerate(data_train):
			batch_data = (Variable(t).to(device) for t in batch_data)
			x, y, lens = batch_data
			optimizer.zero_grad()
			scores, loss, a_acc = model(x, y, lens)
			loss.backward()
			optimizer.step()
			batches_loss += loss.item()
			batches_acc += a_acc
			if (num_iter + 1) % 400 == 0:
				batches_loss /= 400
				batches_acc /= 400
				print('****************************************')
				print(" * Iteration: {}/{} Epoch: {}/{}".format(num_iter + 1, len(data_train), epoch, sp.num_epoch))
				print(" * loss\t {}\t accuracy\t {}\n".format(round(batches_loss, 4), round(batches_acc, 4)))
				batch_train_loss.append(batches_loss)
				batch_train_accuracy.append(batches_acc)
				batches_loss = batches_acc = 0
			if (num_iter + 1) % 500 == 0:
				model.save()

		model.eval()
		start = token2id.get("^", 37)
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
		print("\n * test\tbeam\t{}\tgreedy\t{}".format(sum(beam_pre) / num, sum(greedy_pre) / num))


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
