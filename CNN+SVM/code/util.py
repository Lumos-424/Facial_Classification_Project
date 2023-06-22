import numpy as np
import re
from PIL import Image
import math

# -----------------------------------------------------------------
# 功能：读取标签文件，并将结果保存为一个list
# 输入：label_Path(标签文件路径)
# 输出：label_list(由字典组成的list，详见readme.txt和label.txt)
# 例如：{'id': '5222', 'sex': 'female', 'age': 'adult', 'race': 'white', 'face': 'smiling', 'prop': '', 'missing': 'false'}
# -----------------------------------------------------------------
def get_label_list(label_Path) :
	f1 = open(label_Path+"faceDS","rb")
	f2 = open(label_Path+"faceDR","rb")
	label_1 = f1.read().decode()
	label_2 = f2.read().decode()
	label_3 = label_1 + label_2
	label_3 = label_3.split('\n')  # 这个时候的标签是一个元素为str的list
	label_list = []
	for str in label_3 :
		dict = {'id':'','sex':'','age':'', 'race':'', 'face':'', 'prop':'', 'missing':'false'}
		l = re.findall("\((.*?)\)",str)       # 从一堆字符串中提取想要的信息
		if len(l) == 0:
			continue
		if "_missing descriptor" in l :
			dict['id'] = str[1:5]
			dict['missing'] = 'true'
			label_list.append(dict)
			continue
		dict['sex'] = l[0][6:]
		dict['age'] = l[1][6:]
		dict['race'] = l[2][6:]
		dict['face'] = l[3][6:]
		dict['prop'] = l[4][8:]
		dict['id'] = str[1:5]
		label_list.append(dict)
	return label_list


# ------------------------------------------------------------------------------
# 功能：从所有标签中提取指定的标签并编码
# 输入：label_list(包含所有标签的list)、label_name(想要提取的指定的标签，例如：'sex')
# 输出：返回指定标签编码后的list(由0、1、2.....表示不同的类),分类列表，和分类个数
# label_encoded = [0, 1, 2, 0...]; m=['元素', '',...]
# -------------------------------------------------------------------------------
def encode(label_list, label_name):
	label_encoded = []      # 储存编码后的标签
	label_element = []                  # 储存标签字符串,内容不能重复
	for dict in label_list :
		x = dict[label_name]   # 取出标签字符串
		if x not in label_element:
			label_element.append(x)
		label_encoded.append(label_element.index(x))
	return label_encoded, label_element, len(label_element)


# -----------------------------------------------------------------
# 功能：将图片转化为一个numpy数组
# 输入：img_path(图片路径)、label(标签数组)
# 输出：无返回值，保存img_data(numpy储存的数组)的npy文件,shape为3993*16384
# -----------------------------------------------------------------
def get_img_data(img_path, label_list):
	id = label_list[0]['id']
	f = open(img_path + id, "rb")
	img_data = np.frombuffer(f.read(), dtype=np.uint8)
	for i in range(1, len(label_list)):
		if label_list[i]['missing'] != 'true' :
			id = label_list[i]['id']
			print(id)
			path_img = img_path + id
			f = open(path_img, "rb")
			x = np.frombuffer(f.read(), dtype=np.uint8)
			if x.shape[0] != 128*128:  # 有的图像不是128*128的，把它reshape
				size = x.shape
				print('id:' + str(id) + ' size:' + str(size))
				size = int(math.sqrt(size[0]))
				x = x.reshape((size, size))
				x = Image.fromarray(x)
				x = x.resize((128, 128))
				x = np.array(x)
				x = x.reshape(1, -1)
			img_data = np.vstack((img_data, x))  # 垂直(行)按顺序堆叠数组。
	print(img_data.shape)
	np.save('./ImgData/img_old.npy', img_data)


# -----------------------------------------------------------------
# 功能：把人脸图像保存为jpg文件
# 输入：img_data(图片数据)
# 输出：无
# -----------------------------------------------------------------
def img2jpg(img_data) :
	label_Path = './人脸图像识别/face/'
	save_Path = './facedata/jpg/'
	label_dict = get_label_list(label_Path)

	for i in range(img_data.shape[0]) :
		id = label_dict[i]['id']
		img = img_data[i]
		img = img.reshape(128,128)
		img = Image.fromarray(img)
		img.save(save_Path + id + '.jpg')


# -----------------------------------------------------------------
# 功能：删除错误数据
# 输入：img_data(图片数据)、label(标签数组)
# 输出：返回 img_del 和 label_del，为删除错误数据及标签后的 (3961,16384)
# -----------------------------------------------------------------
def DeleteBadData(img_data, label) :
	img_mean = np.mean(img_data,axis=1) # 计算图片均值(一张图片所有像素值的均值)

	index = np.where(img_mean < 1 )
	index = index[0] # index是一个元组，[0]提取出数据
	id_abnormal = []
	for i in index :
		l = label[i]
		id = l['id']
		id_abnormal.append(id)
	print('数据错误图片对应的id为：',id_abnormal)
	img_del = np.delete(img_data,index,0)
	label_del = [n for i, n in enumerate(label) if i not in index]
	return img_del,label_del


# -----------------------------------------------------------------
# 功能：划分数据集变为训练集、验证集、测试集
# -----------------------------------------------------------------
import os
import random
from shutil import copy2

def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):

	'''
	读取源数据文件夹，生成划分好的文件夹，分为trian、valid、test三个文件夹进行
	:param src_data_folder: 源文件夹
	:param target_data_folder: 目标文件夹
	:param train_scale: 训练集比例
	:param val_scale: 验证集比例
	:param test_scale: 测试集比例
	:return:
	'''

	print("开始数据集划分")
	class_names = os.listdir(src_data_folder)
	# 在目标目录下创建文件夹
	split_names = ['train', 'valid', 'test']
	for split_name in split_names:
		split_path = os.path.join(target_data_folder, split_name)
		if os.path.isdir(split_path):
			pass
		else:
			os.mkdir(split_path)
		# 然后在split_path的目录下创建类别文件夹
		for class_name in class_names:
			class_split_path = os.path.join(split_path, class_name)
			if os.path.isdir(class_split_path):
				pass
			else:
				os.mkdir(class_split_path)

	# 按照比例划分数据集，并进行数据图片的复制
	# 首先进行分类遍历
	for class_name in class_names:
		current_class_data_path = os.path.join(src_data_folder, class_name)
		current_all_data = os.listdir(current_class_data_path)
		current_data_length = len(current_all_data)
		current_data_index_list = list(range(current_data_length))
		random.shuffle(current_data_index_list)

		train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
		val_folder = os.path.join(os.path.join(target_data_folder, 'valid'), class_name)
		test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
		train_stop_flag = current_data_length * train_scale
		val_stop_flag = current_data_length * (train_scale + val_scale)
		current_idx = 0
		train_num = 0
		val_num = 0
		test_num = 0
		for i in current_data_index_list:
			src_img_path = os.path.join(current_class_data_path, current_all_data[i])
			if current_idx <= train_stop_flag:
				copy2(src_img_path, train_folder)
				# print("{}复制到了{}".format(src_img_path, train_folder))
				train_num = train_num + 1
			elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
				copy2(src_img_path, val_folder)
				# print("{}复制到了{}".format(src_img_path, val_folder))
				val_num = val_num + 1
			else:
				copy2(src_img_path, test_folder)
				# print("{}复制到了{}".format(src_img_path, test_folder))
				test_num = test_num + 1

			current_idx = current_idx + 1

		print("*********************************{}*************************************".format(class_name))
		print(
			"{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale, current_data_length))
		print("训练集{}：{}张".format(train_folder, train_num))
		print("验证集{}：{}张".format(val_folder, val_num))
		print("测试集{}：{}张".format(test_folder, test_num))




