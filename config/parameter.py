#!/usr/bin python3
# -*- coding: utf-8 -*-
# @时间   : 19-4-15 上午11:05
# @作者   : Lin lifang
# @文件   : parameter.py

MIN_HEIGHT = 40
MIN_WIDTH = 100

MIN_LEN = 4
MAX_LEN = 7

ImageHeight = 48
ImageWidth = 128

VOCAB_SIZE = 37


class CTCParam:
	BATCH_SIZE = 200
	num_epoch = 10
	num_layer = 1
	num_units = 100
	dropout = 0.5
	clip = 5
	lr = 5e-4


token2id = {
	"_": 0,
	"0": 1,
	"1": 2,
	"2": 3,
	"3": 4,
	"4": 5,
	"5": 6,
	"6": 7,
	"7": 8,
	"8": 9,
	"9": 10,
	"a": 11,
	"b": 12,
	"c": 13,
	"d": 14,
	"e": 15,
	"f": 16,
	"g": 17,
	"h": 18,
	"i": 19,
	"j": 20,
	"k": 21,
	"l": 22,
	"m": 23,
	"n": 24,
	"o": 25,
	"p": 26,
	"q": 27,
	"r": 28,
	"s": 29,
	"t": 30,
	"u": 31,
	"v": 32,
	"w": 33,
	"x": 34,
	"y": 35,
	"z": 36,
	"A": 37,
	"B": 38,
	"C": 39,
	"D": 40,
	"E": 41,
	"F": 42,
	"G": 43,
	"H": 44,
	"I": 45,
	"J": 46,
	"K": 47,
	"L": 48,
	"M": 49,
	"N": 50,
	"O": 51,
	"P": 52,
	"Q": 53,
	"R": 54,
	"S": 55,
	"T": 56,
	"U": 57,
	"V": 58,
	"W": 59,
	"X": 60,
	"Y": 61,
	"Z": 62,
	"^": 63,
	"$": 64
}
