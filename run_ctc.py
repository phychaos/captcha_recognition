#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午10:51
# @作者   : Lin lifang
# @文件   : run_ctc.py
import os
from tqdm import tqdm

from config.config import TEST_DATA
from config.parameter import VOCAB_SIZE
from core.utils import load_dataset, load_image
from models.ctc_model import CTCModel
from config.parameter import CTCParam as hp
from torch.optim import Adam
from torch.nn import CTCLoss
import torch
from torch.autograd import Variable
from config.parameter import token2id

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def run():
	model = CTCModel(VOCAB_SIZE, hp.num_layer, hp.hidden_size, hp.dropout, use_cuda)
	model.load_model()
	if use_cuda:
		model.cuda()

	params = list(filter(lambda p: p.requires_grad, model.parameters()))
	optimizer = Adam(params, lr=hp.lr)
	criterion = CTCLoss()

	data_train, data_test = load_dataset(batch_size=hp.BATCH_SIZE)
	train_num = len(data_train)
	test_num = len(data_test)
	for epoch in range(1, hp.num_epoch + 1):
		batches_loss = batches_acc = 0
		model.train()
		for num_iter, batch_data in enumerate(tqdm(data_train, desc="训练")):
			x, y, _, seq_lens = batch_data
			y = Variable(y[y > 0].contiguous()).to(device)
			x = Variable(x).to(device)
			seq_lens = Variable(seq_lens).to(device)

			optimizer.zero_grad()
			outputs = model(x)
			acc, _ = model.accuracy(outputs, y, seq_lens, blank=0)

			batch_size, seq_len, _ = outputs.size()
			act_lens = Variable(torch.IntTensor(batch_size * [seq_len]), requires_grad=False)
			loss = criterion(outputs.transpose(0, 1), y, act_lens, seq_lens)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip)
			optimizer.step()

			batches_loss += loss.item()
			batches_acc += acc
		batches_loss /= train_num
		batches_acc /= train_num
		model.save()
		model.eval()

		# test
		loss = accuracy = 0
		with torch.no_grad():
			for num_iter, batch_data in enumerate(tqdm(data_test, "测试")):
				x, y, _, seq_lens = batch_data
				y = Variable(y[y > 0].contiguous()).to(device)
				x = Variable(x).to(device)
				seq_lens = Variable(seq_lens).to(device)
				outputs = model(x)
				a_loss, _ = model.accuracy(outputs, y, seq_lens, blank=0)

				batch_size, seq_len, _ = outputs.size()
				act_lens = Variable(torch.IntTensor(batch_size * [seq_len]), requires_grad=False)
				a_loss = criterion(outputs.transpose(0, 1), y, act_lens, seq_lens)
				acc, _ = model.accuracy(outputs, y, seq_lens, blank=0)
				loss += a_loss.item()
				accuracy += acc
		loss = loss / test_num
		accuracy = accuracy / test_num
		print('\n****************************************')
		print(" * Epoch: {}/{}".format(epoch, hp.num_epoch))
		print(" * loss\t {}\t accuracy\t {}".format(round(batches_loss, 4), round(batches_acc, 4)))
		print(" * test loss:\t{}\taccuracy:\t{}".format(round(loss, 4), round(accuracy, 4)))


def test():
	model = CTCModel(output_size=VOCAB_SIZE, num_layers=hp.num_layer, hidden_size=hp.hidden_size, dropout=hp.dropout)
	model.load_model()
	id2token = {str(idx): token for token, idx in token2id.items()}
	for filename in os.listdir(TEST_DATA):
		x, y, lens, label = load_image(TEST_DATA + filename)
		x = Variable(x.unsqueeze(0))
		outputs = model(x)
		predict = model.decode_ctc_outputs(outputs, blank=0)
		pre_label = ''.join([id2token.get(str(idx), '_') for idx in predict[0]])
		acc = 1 if pre_label.lower() == label.lower() else 0
		print("predict:\t{}\t\tlabel:\t{}\t\t{}".format(pre_label, label, acc))


if __name__ == '__main__':
	run()
	# test()
