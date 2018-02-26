#coding:utf-8
"""
Author:wepon
Source:https://github.com/wepe
"""


import os
from PIL import Image
import numpy as np

#读取eeg件夹
def load_data(img_rows, img_cols,channel, path):
	imgs = os.listdir(path)
	num = len(imgs)

	data = np.empty((num, channel, img_rows, img_cols),dtype="float32")
	label = np.empty((num),dtype="uint8")
	
	for i in range(num):

		eegData = np.loadtxt(path+"/"+imgs[i])
		# for j in range(3749):
		# 	eegData[j] = eegData[j+1] - eegData[j]
		# eegData[3749] = eegData[3748]

		eegLabel = int(imgs[i].split('-')[0])

		data[i,:,:,:] = eegData.reshape(1, channel, img_rows,img_cols)
		label[i] = eegLabel
		
	recordOrder = np.empty(num, dtype='<U32')
	for i in range(num):
		recordOrder[i] = imgs[i]
	return data,label,recordOrder
