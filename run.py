#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午10:51
# @作者   : Lin lifang
# @文件   : run.py
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config.config import *
from config.parameter import VOCAB_SIZE
from core.gen_captcha import get_captcha, convert_to_npz
from core.utils import load_dataset
from models.ctc_model import CTCModel, CTCtrain, CTCevaluate
from config.parameter import CTCParam as hp
from torch.optim import Adam
from torch.nn import CTCLoss
import torch
from torch.autograd import Variable
from config.parameter import token2id


def run():
	ctc = CTCModel(output_size=VOCAB_SIZE, num_layers=hp.num_layer, num_units=hp.num_units, dropout=hp.dropout)
	ctc.load_model()
	use_cuda = False
	device = torch.device("cuda" if use_cuda else "cpu")
	if use_cuda:
		ctc.cuda()

	ctc_params = list(filter(lambda p: p.requires_grad, ctc.parameters()))
	ctc_optimizer = Adam(ctc_params, lr=hp.lr)
	criterion = CTCLoss()

	data_train, data_test = load_dataset(batch_size=hp.BATCH_SIZE)

	batch_train_loss = []
	batch_train_accuracy = []
	for epoch in range(1, hp.num_epoch + 1):
		batches_loss = batches_acc = 0
		for num_iter, batch_data in enumerate(tqdm(data_train, desc="训练集")):
			x, y, lens = batch_data
			y = Variable(y[y > 0].contiguous()).to(device)
			x = Variable(x).to(device)
			lens = Variable(lens).to(device)

			a_loss, a_accuracy = CTCtrain(x, y, lens, ctc, ctc_optimizer, criterion, hp.clip, use_cuda=use_cuda)
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
				ctc.save(str(epoch) + "_" + str(num_iter + 1))

		# test
		loss = accuracy = 0
		for num_iter, batch_data in enumerate(tqdm(data_test, "测试集")):
			x, y, lens = batch_data
			y = Variable(y[y > 0].contiguous()).to(device)
			x = Variable(x).to(device)
			lens = Variable(lens).to(device)
			a_loss, a_accuracy, outputs = CTCevaluate(x, y, lens, ctc, criterion, hp.clip, use_cuda=use_cuda)
			loss += a_loss
			accuracy += a_accuracy
		loss = loss / len(data_test)
		accuracy = accuracy / len(data_test)
		print("\n * test loss:\t{}\taccuracy:\t{}".format(round(loss, 4), round(accuracy, 4)))


def gen_image():
	# get_captcha(num=200000, path=TRAIN_DATA)
	# get_captcha(num=1000, path=TEST_DATA)
	get_captcha(num=10, path=IMAGE_DATA)


if __name__ == '__main__':
	gen_image()
# run()
