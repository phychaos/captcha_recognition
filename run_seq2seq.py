#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-16 下午5:11
# @作者   : Lin lifang
# @文件   : run_seq2seq.py
import os
from tqdm import tqdm
from config.parameter import VOCAB_SIZE, MAX_LEN
from core.utils import load_dataset, load_image
from config.parameter import CTCParam as hp
from torch.optim import Adam
from torch.nn import CTCLoss
import torch
from torch.autograd import Variable
from config.parameter import token2id
from models.seq2seq_model import Seq2seqModel, train, evaluate, seq2seq_test
from config.parameter import Seq2seqParam as sp


def run():
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	vocab_size = VOCAB_SIZE + 2
	model = Seq2seqModel(sp.attn_model, vocab_size, sp.hidden_size, vocab_size, sp.num_layer, sp.dropout)
	model.load_model()
	if use_cuda:
		model.cuda()

	params = list(filter(lambda p: p.requires_grad, model.parameters()))
	optimizer = Adam(params, lr=hp.lr)
	criterion = torch.nn.CrossEntropyLoss()

	data_train, data_test = load_dataset(batch_size=hp.BATCH_SIZE, model="seq2seq")

	batch_train_loss = []
	batch_train_accuracy = []
	for epoch in range(1, hp.num_epoch + 1):
		batches_loss = batches_acc = 0
		for num_iter, batch_data in enumerate(tqdm(data_train, desc="训练集")):
			batch_data = (Variable(t).to(device) for t in batch_data)
			x, y, lens = batch_data

			a_loss, a_accuracy = train(x, y, model, optimizer, criterion, sp.hidden_size, sp.clip, use_cuda=use_cuda)
			batches_loss += a_loss
			batches_acc += a_accuracy

			if (num_iter + 1) % 200 == 0:
				batches_loss /= 200
				batches_acc /= 200
				print('\n****************************************')
				print(" * Iteration: {}/{} Epoch: {}/{}".format(num_iter + 1, len(data_train), epoch, hp.num_epoch))
				print(" * loss\t {}\t accuracy\t {}\n".format(round(batches_loss, 4), round(batches_acc, 4)))
				batch_train_loss.append(batches_loss)
				batch_train_accuracy.append(batches_acc)
				batches_loss = batches_acc = 0
			if (num_iter + 1) % 500 == 0:
				model.save(str(epoch) + "_" + str(num_iter + 1))

		# test
		loss = accuracy = 0
		for num_iter, batch_data in enumerate(tqdm(data_test, "测试集")):
			batch_data = (Variable(t).to(device) for t in batch_data)
			x, y, lens = batch_data
			a_loss, a_accuracy, outputs = evaluate(x, y, model, criterion, sp.hidden_size, use_cuda=use_cuda)
			loss += a_loss
			accuracy += a_accuracy
		loss = loss / len(data_test)
		accuracy = accuracy / len(data_test)
		print("\n * test loss:\t{}\taccuracy:\t{}".format(round(loss, 4), round(accuracy, 4)))


def test():
	vocab_size = VOCAB_SIZE + 2
	model = Seq2seqModel(sp.attn_model, vocab_size, sp.hidden_size, vocab_size, sp.num_layer, sp.dropout)
	model.load_model()

	id2token = {str(idx): token for token, idx in token2id.items()}
	for filename in os.listdir('./images'):
		x, y, lens, label = load_image('./images/' + filename)
		x = Variable(x.unsqueeze(0))
		target = Variable(torch.LongTensor([[token2id.get('^', 37)]]))
		outputs = seq2seq_test(x, target, MAX_LEN + 2, model, sp.hidden_size, use_cuda=False)
		pre_label = ''.join([id2token.get(str(idx), '_') for idx in outputs])
		pre_label = pre_label.split('$')[0]
		acc = 1 if pre_label.lower() == label.lower() else 0
		print("pre:\t{}\t\ttruth:\t{}\t\t{}".format(pre_label, label, acc))


if __name__ == '__main__':
	run()
	test()
