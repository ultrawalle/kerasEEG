'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from data import load_data
# from data_fileNameList import load_dataAndRecordOrder
import random
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import os
import os.path

# from sklearn.model_selection import StratifiedKFold



class model_IMF(object):
	def __init__(self,npDataPath =	None,trainDataPath=None,img_rows = 22,img_cols = 3000,channel = 1,batch_size = 128,num_classes =5,epochs =30):
		self.npDataPath = npDataPath
		self.trainDataPath = trainDataPath
		self.img_rows = img_rows
		self.img_cols = img_cols
		self.channel = channel
		self.batch_size = batch_size
		self.num_classes = num_classes
		self.epochs = epochs
		self.input_shape = (self.channel, self.img_rows, self.img_cols)
		np.random.seed(1024)
		self.lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
		self.early_stopper = EarlyStopping(min_delta=0.001, patience=10)
		self.csv_logger = CSVLogger(npDataPath + 'resnet18_cifar10.csv')
		if self.trainDataPath != None:
			self.load_txt(trainDataPath)
		elif self.npDataPath!=None:
			self.load_np(npDataPath)

  # for reproducibility

# batch_size = 128
	def load_txt(self,trainDataPath):
		npDataPath = self.npDataPath
		self.x_train, self.y_train,  self.recordOrder= load_data(self.img_rows, self.img_cols, self.channel, trainDataPath)
		# self.x_test1, self.y_test1, recordOrder = load_dataAndRecordOrder(img_rows, img_cols, channel, testDataPath)

		for x in range(len(self.y_train)):
			if self.y_train[x] == 4:
				self.y_train[x] = 3

		for x in range(len(self.y_train)):
			if self.y_train[x] == 5:
				self.y_train[x] = 4


		#shuffle data
		index = [i for i in range(len(self.x_train))]
		random.shuffle(index)

		self.recordOrder = self.recordOrder[index]

		recordRateIndex = int(0.7*self.x_train.shape[0])
		self.x_train = self.x_train[index]
		self.y_train = self.y_train[index]

		self.x_test = self.x_train[recordRateIndex:]
		self.y_test = self.y_train[recordRateIndex:]
		self.recordOrder = self.recordOrder[recordRateIndex:]

		self.x_train = self.x_train[0:recordRateIndex]
		self.y_train = self.y_train[0:recordRateIndex]

		# npDataPath = "/data/weiliangjie/PhysioNetDataset/data/npData/"
		np.save(npDataPath + "x_train.npy", self.x_train)
		np.save(npDataPath + "x_test.npy", self.x_test)
		np.save(npDataPath + "y_train.npy", self.y_train)
		np.save(npDataPath + "y_test.npy", self.y_test)
		np.savetxt(npDataPath + "y_test.txt", self.y_test)
		np.savetxt(npDataPath + "recordOrder.txt", self.recordOrder, fmt='%s')
		self.status_and_pre()
 

	def load_np(self,npDataPath):
		
		self.x_train = np.load(npDataPath + "x_train.npy")
		self.x_test = np.load(npDataPath + "x_test.npy")
		self.y_train = np.load(npDataPath + "y_train.npy")
		self.y_test = np.load(npDataPath + "y_test.npy")
		self.recordOrder = np.loadtxt(npDataPath + "recordOrder.txt", dtype=bytes).astype(str)
		self.status_and_pre()

	def status_and_pre(self):
		self.x_train /= 10
		self.x_test /= 10

		print('self.x_train shape:', self.x_train.shape)
		print('self.x_test shape:', self.x_test.shape)

		print(self.x_train.shape[0], 'train samples')
		print(self.x_test.shape[0], 'test samples')

		print("\n")

		trainStage0 = np.sum(self.y_train == 0)
		testStage0 = np.sum(self.y_test == 0)

		trainStage1 = np.sum(self.y_train == 1)
		testStage1 = np.sum(self.y_test == 1)

		trainStage2 = np.sum(self.y_train == 2)
		testStage2 = np.sum(self.y_test == 2)

		trainStage3 = np.sum(self.y_train == 3)
		testStage3 = np.sum(self.y_test == 3)

		trainStage5 = np.sum(self.y_train == 4)
		testStage5 = np.sum(self.y_test == 4)

		print('stage0: ', trainStage0, ',', testStage0)
		print('stage1: ', trainStage1, ',', testStage1)
		print('stage2: ', trainStage2, ',', testStage2)
		print('stage3: ', trainStage3, ',', testStage3)
		print('stage5: ', trainStage5, ',', testStage5)
		print("\n")
		#-----------------------------------------------------------

		self.x_train = self.x_train.astype('float32')
		self.x_test = self.x_test.astype('float32')
		self.y_test_copy = self.y_test
		# convert class vectors to binary class matrices
		self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
		self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)	
		self.train_model()

	def train_model(self):
		self.model = Sequential()

		# self.model.add(Conv2D(128, kernel_size=(2, 15), strides=(1, 5),
		# activation='relu',
		# input_shape=self.input_shape))
		self.model.add(Conv2D(128, kernel_size=(3,3),
		activation='relu',
		input_shape=self.input_shape))

		# model.add(Conv2D(128, kernel_size=(3, 3),
		#                  activation='relu',
		#                  input_shape=input_shape))

		self.model.add(Conv2D(128, (3, 3), activation='relu'))
		self.model.add(Dropout(0.25))
		# model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
		self.model.add(AveragePooling2D(pool_size=(2, 2)))


		self.model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
		#*************************************************************************************************

		self.model.add(Flatten())
		self.model.add(Dense(300, activation='relu'))
		# model.add(Dense(200, activation='relu'))
		self.model.add(Dropout(0.25))
		self.model.add(Dense(self.num_classes, activation='softmax'))

		self.model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy'])

	# early_stopping = EarlyStopping(monitor='val_loss', patience=2)

	# model.fit(self.x_train, self.y_train,
	#           batch_size=batch_size,
	#           epochs=epochs,
	#           verbose=1,
	#           validation_data=(self.x_test, self.y_test),
	#           callbacks=[early_stopping])


	# model.fit(self.x_train, self.y_train,
	#           batch_size=batch_size,
	#           epochs=epochs,
	#           verbose=1,
	#           validation_data=(self.x_test, self.y_test),
	#           callbacks=[lr_reducer, early_stopper, csv_logger])

		self.model.fit(self.x_train, self.y_train,
						  batch_size=self.batch_size,
						  epochs=self.epochs,
						  verbose=1,
						  shuffle=True,
						  validation_split=0.2,
						  callbacks=[self.lr_reducer, self.early_stopper, self.csv_logger])
		self.save_model(self.npDataPath)


	def save_model(self,npDataPath):
		
		if os.path.exists(npDataPath + 'my_model_architecture.json'):
			os.remove(npDataPath + 'my_model_architecture.json')
		if os.path.exists(npDataPath + 'my_model_weights.h5'):
			os.remove(npDataPath + 'my_model_weights.h5')
		#---------------------------------------------------保存模型-------------------------------------------------

		json_string = self.model.to_json()
		open(npDataPath + 'my_model_architecture.json','w').write(json_string)
		self.model.save_weights(npDataPath + 'my_model_weights.h5') 	
		print('self.x_test shape:', self.x_test.shape)
		score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

		#########################下面还没改

		preRes = self.model.predict_classes(self.x_test)

		# -----------------------计算预测结果准确率------------------------------------
		result = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

		num = len(self.y_test_copy)
		y_real = np.empty((num), dtype="uint8")
		y_preResult = np.empty((num), dtype="uint8")

		for i in range(len(self.y_test_copy)):
			y = int(self.y_test_copy[i])
			y_pre = int(preRes[i])
			result[y][y_pre] = result[y][y_pre] + 1
			if y == 4:
				y = 5
			if y_pre == 4:
				y_pre = 5
			y_real[i] = y
			y_preResult[i] = y_pre

		print('******************************** 实验结果分析 **************************************')
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		print('\n')

		mf1 = 0.0

		for i in range(0, 5, 1):
			total = sum(result[i])
			stageLabel = ''
			if i == 4:
				stageLabel = 'Rem   ：'
			else:
				stageLabel = 'stage' + str(i) + '：'
			print('******************************** ', stageLabel, ' **************************************')
			print(stageLabel, '-->total number: ', total)
			coli = list((result[j][i] for j in range(5)))
			coli_num_total = sum(coli)

			print('Test accuracy: ', result[i][i] / coli_num_total, '%')
			print('Test recall  : ', result[i][i] / total, '%')

			print('Test F1-Score: ', 2 * ((result[i][i] / coli_num_total) * (result[i][i] / total)) / (
			(result[i][i] / coli_num_total) + (result[i][i] / total)), '%')
			mf1 = mf1 + 2 * ((result[i][i] / coli_num_total) * (result[i][i] / total)) / (
			(result[i][i] / coli_num_total) + (result[i][i] / total))

			print(stageLabel, ' --> ', 'stage0：', total, ' --> ', result[i][0])
			print(stageLabel, ' --> ', 'stage1：', total, ' --> ', result[i][1])
			print(stageLabel, ' --> ', 'stage2：', total, ' --> ', result[i][2])
			print(stageLabel, ' --> ', 'stage3：', total, ' --> ', result[i][3])
			print(stageLabel, ' --> ', 'Rem   ：', total, ' --> ', result[i][4])

		print('******************************** MF1 **************************************')
		mf1 = mf1 / 5
		print('MF1  : ', mf1, '%')
		print('******************************** MF1 **************************************')

		# -----------------------------------------------------------------------------
		result = np.empty((num), dtype='<U32')
		for i in range(num):
			tmp = str(y_real[i]) + ',' + str(y_preResult[i]) + ',' + str(self.recordOrder[i])
			result[i] = tmp

		if os.path.exists(npDataPath + 'result.csv'):
			os.remove(npDataPath + 'result.csv')
		if os.path.exists(npDataPath + 'self.y_test_preRes.txt'):
			os.remove(npDataPath + 'self.y_test_preRes.txt')

		np.savetxt(npDataPath + "result.csv", result, fmt='%s')
		np.savetxt(npDataPath + "y_test_preRes.txt", preRes)






##################
# #npDataPath = "/data/weiliangjie/PhysioNetDataset/dataOfTime/eegSourceData_20170628_npData_0.2/"
# npDataPath = "/data/wanglei/EEGProject/dataOfIMF/EMD_IMF_20170719/np/"
# #npDataPath = "/data/wanglei/EEGProject/dataOfIMF/EEMD_IMF_20170719/np/"



# # npDataPath = "/data/weiliangjie/PhysioNetDataset/data/npData/"




# batch_size = 128
# num_classes = 5
# epochs = 30

# # input image dimensions
# # img_rows, img_cols, channel = 750, 750, 3
# img_rows, img_cols, channel = 22, 3000, 1
# input_shape = (channel, img_rows, img_cols)
#################


# #---------------------试验一--------------------------------------------------------------
# #保存数据

if __name__ == '__main__':
	# trainDataPath="/data/wanglei/EEGProject/dataOfIMF/EMD_IMF_20170719/EMD_IMF"
	# npDataPath = "/data/wanglei/EEGProject/dataOfIMF/EMD_IMF_20170719/np/"
	# model_imf = model_IMF(trainDataPath = None,npDataPath = npDataPath)

	# trainDataPath="/data/wanglei/EEGProject/dataOfIMF/EEMD_IMF_20170719/EEMD_IMF"
	# npDataPath = "/data/wanglei/EEGProject/dataOfIMF/EEMD_IMF_20170719/np/"
	# model_imf = model_IMF(trainDataPath = None,npDataPath = npDataPath)

	trainDataPath="/data/WangLei/wanglei/EEGProject/dataOfSpectrogramReduceNotN1/EMD"
	npDataPath = "/data/WangLei/wanglei/EEGProject/dataOfSpectrogramReduceNotN1/EMD_np"
	model_imf = model_IMF(trainDataPath = trainDataPath,npDataPath = npDataPath, img_rows = 40, img_cols=30)

	trainDataPath="/data/WangLei/wanglei/EEGProject/dataOfSpectrogramReduceNotN1/EEMD"
	npDataPath = "/data/WangLei/wanglei/EEGProject/dataOfSpectrogramReduceNotN1/EEMD_np"
	model_imf = model_IMF(trainDataPath = trainDataPath,npDataPath = npDataPath, img_rows = 40, img_cols=30)


#trainDataPath="/data/wanglei/EEGProject/dataOfIMF/EEMD_IMF_20170719/txt"

# #---------------------------------------------------------------------------------------------

#---------------------试验2--------------------------------------------------------------
#加载数据

#npDataPath = "/data/weiliangjie/PhysioNetDataset/data/npData/"












# # # #---------------------------------------------------------------------------------------------



# -------------------------------------------*****一*****-----------------------------------------------------------
# model = Sequential()

# # model.add(Conv2D(128, kernel_size=(2, 10), strides=(1, 3),
# #                  activation='relu',
# #                  input_shape=input_shape))

# model.add(Conv2D(128, kernel_size=(2, 5),
#                  activation='relu',
#                  input_shape=input_shape))

# model.add(Conv2D(64, (3, 3), activation='relu'))
# # model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
# # model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))

# model.add(Conv2D(16, kernel_size=(3, 3),activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))

#*****************************************二*****************************************************


#---------------------------------------------------保存模型-------------------------------------------------








