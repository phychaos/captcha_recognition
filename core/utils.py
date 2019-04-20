#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午11:23
# @作者   : Lin lifang
# @文件   : utils.py
import numpy as np
from torch.utils import data
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from config.config import TRAIN_NPZ, TEST_NPZ, TRAIN_DATA, TEST_DATA
from config.parameter import ImageHeight, ImageWidth, MAX_LEN, token2id
import re

p = re.compile('_|\.')


def decode_ctc_outputs(ctc_outputs, blank=0):
    outputs = ctc_outputs.max(dim=-1)[1].cpu().data.numpy()
    seq_len = outputs.shape[1]
    imres = [
        np.array([sample[i] for i in range(seq_len - 1) if sample[i] != sample[i - 1]] + [sample[-1]], dtype=np.int32)
        for sample in outputs]
    return [sample[sample != blank] for sample in imres]


class Captcha(data.Dataset):
    def __init__(self, root='/', model='ctc'):
        self.model = model
        self.images_path = [os.path.join(root, img) for img in os.listdir(root)]
        self.transform = transforms.Compose([
            transforms.Resize((ImageHeight, ImageWidth)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = p.split(img_path.split("/")[-1])[0]
        y, label_len = str2label(label, self.model)
        y = torch.LongTensor(y)
        image = Image.open(img_path)
        image = self.transform(image)
        return image, y, label_len

    def gen_image(self, filename):
        label = p.split(filename.split("/")[-1])[0]
        y, label_len = str2label(label)
        y = torch.Tensor(y).type(torch.IntTensor)
        image = Image.open(filename)
        image = self.transform(image)
        return image, y, label_len, label

    def __len__(self):
        return len(self.images_path)


def str2label(str_label, model='ctc'):
    """字符串转ID"""
    label = []
    label_len = len(str_label)
    if model == 'ctc':
        str_label = str_label.ljust(MAX_LEN, "_")
    else:
        str_label = '^' + str_label + '$'
        str_label = str_label.ljust(MAX_LEN + 2, "_")
        label_len += 2

    for kk in str_label:
        idx = token2id.get(kk, 0)
        label.append(idx)
    return label, label_len


def load_npz(filename):
    npz_data = np.load(filename)
    image = npz_data["img"].astype(np.float32) / 127.5 - 1
    text = npz_data["text"]
    image_len = npz_data["image_len"]
    image = image.transpose(0, 3, 1, 2)
    return image, text, image_len


def load_dataset(batch_size, model="ctc"):
    """
    加载数据
    :param batch_size:
    :param model:
    :return:
    """
    train_data_set = Captcha(TRAIN_DATA, model)
    test_data_set = Captcha(TEST_DATA, model)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_data_loader, test_data_loader


def generate_batch_data(x, y, z, batch_size=100):
    num_sample = len(z)
    total_batch = num_sample // batch_size
    if num_sample % batch_size > 0:
        total_batch += 1
    data_load = []
    for ii in range(total_batch):
        start, end = ii * batch_size, (ii + 1) * batch_size
        x_data = x[start:end, :, :, :]
        y_data = y[start:end]
        z_data = z[start:end]
        data_load.append([x_data, y_data, z_data])
    return data_load


def load_image(filename):
    data_captcha = Captcha()
    images, y, label_len, label = data_captcha.gen_image(filename)
    return images, y, label_len, label
