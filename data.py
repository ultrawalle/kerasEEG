#coding:utf-8
"""
Author:wepon
Source:https://github.com/wepe
"""


import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，图像大小28*28
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
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