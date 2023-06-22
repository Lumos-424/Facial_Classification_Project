import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.manifold import Isomap
from PIL import Image
import math
import random
# 查看图像
# f = open("FR/face/rawdata/4359","rb")
# x = np.frombuffer(f.read(),dtype=np.uint8)
# x = x.reshape(128,128)
# plt.imshow(x,cmap='gray')
# plt.show()

#-----------------------------------------------------------------
#功能：读取标签文件，并将结果保存为一个list
#输入：label_Path(标签文件路径)
#输出：label(由字典组成的list，详见readme.txt和label.txt)
#-----------------------------------------------------------------
def Get_label(label_Path) :
	f1 = open(label_Path+"faceDS","rb")
	f2 = open(label_Path+"faceDR","rb")
	label_1 = f1.read().decode()
	label_1 = label_1.split('\n')
	label_2 = f2.read().decode()
	label_2 = label_2.split('\n')
	label_3 = label_1 + label_2    #这个时候的标签是一个元素为字符串的list
	label = []
	for str in label_3 :
		dict = {'id':'','sex':'','age':'', 'race':'', 'face':'', 'prop':'', 'missing':'false'}
		l = re.findall("\((.*?)\)",str)       #从一堆字符串中提取想要的信息
		if len(l) == 0:
			continue
		if "_missing descriptor" in l :
			dict['id'] = str[1:5]
			dict['missing'] = 'true'
			label.append(dict)
			continue
		dict['sex'] = l[0][6:]
		dict['age'] = l[1][6:]
		dict['race'] = l[2][6:]
		dict['face'] = l[3][6:]
		dict['prop'] = l[4][8:]
		dict['id'] = str[1:5]
		label.append(dict)
	return label


#------------------------------------------------------------------------------
#功能：从所有标签中提取指定的标签并编码
#输入：label_list(包含所有标签的list)、label_name(想要提取的指定的标签，例如：'sex')
#输出：返回指定标签编码后的list(由0、1、2.....表示不同的类),和分类个数
#-------------------------------------------------------------------------------
def encode(label_list, label_name):
	label_encoded = []      #储存编码后的标签
	m = []                  #储存标签字符串,内容不能重复
	for dict in label_list :
		x = dict[label_name]   #取出标签字符串
		if x not in m:         
			m.append(x)
		label_encoded.append(m.index(x)) 
	return label_encoded, len(m)



#-----------------------------------------------------------------
#功能：将图片转化为一个numpy数组
#输入：img_path(图片路径)、label(标签数组)
#输出：无返回值，保存img_data(numpy储存的数组)的npy文件,shape为3993*16384
#-----------------------------------------------------------------
def Get_img(img_path, label) :
	id = label[0]['id']
	f = open(img_path + id,"rb")
	img_data = np.frombuffer(f.read(),dtype=np.uint8)
	for i in range (1,len(label)) :
		if label[i]['missing'] != 'true' :
			id = label[i]['id']
			path_img = img_path + id
			f = open(path_img,"rb")
			x = np.frombuffer(f.read(),dtype=np.uint8)
			if x.shape[0] != 16384 :  #有的图像不是128*128的，把它reshape
				size = x.shape
				size = int(math.sqrt(size[0]))
				x = x.reshape((size,size))
				x = Image.fromarray(x)
				x = x.resize((128,128))
				x = np.array(x)
				x = x.reshape(1,-1)
			img_data = np.vstack((img_data, x))
	print('原始图片数组shape为:',img_data.shape)
	np.save('data/img.npy',img_data)
	return img_data

#-----------------------------------------------------------------
#功能：把人脸图像保存为jpg文件
#输入：img_data(图片数据)、save_Path(jpg保存路径)、label_all(剔除错误数据后的标签字典)
#输出：无
#-----------------------------------------------------------------
def img2jpg(img_data,save_Path,label_all) :
	for i in range(img_data.shape[0]) :
		id = label_all[i]['id']
		img = img_data[i]
		img = img.reshape(128,128)
		img = Image.fromarray(img)
		img.save(save_Path + id + '.jpg')

#-----------------------------------------------------------------
#功能：删除错误数据
#输入：img_data(图片数据)、label(标签数组)
#输出：返回 img_del 和 label_del，为删除错误数据及标签后的
#-----------------------------------------------------------------
def DeleteBadData(img_data, label) :
	img_mean = np.mean(img_data,axis=1) #计算图片均值(一张图片所有像素值的均值)

	index = np.where((img_mean < 1) | (img_mean > 220))  
	index = index[0] #index是一个元组，[0]提取出数据
	id_abnormal = []
	for i in index :
		l = label[i]
		id = l['id']
		id_abnormal.append(id)
	print('数据错误图片对应的id为：',id_abnormal)
	img_del = np.delete(img_data,index,0)
	label_del = [n for i, n in enumerate(label) if i not in index]
	return img_del,label_del

def train_test_split(img_data,label,label_name,ratio,balance = False):
	print('正在划分数据集（该过程将持续3min）：.....')
	train_label, test_label = [],[]
	train_data, test_data = np.zeros((1,16384)),np.zeros((1,16384))
	duplicate_data = np.zeros((1,16384))
	duplicate_label = []
	count = {}
	size = len(label)
	x = [i for i in range(size)]
	random.shuffle(x)
	if label_name == 'sex' :
		count = {'female':1566,'male':2390} 
	if label_name == 'age' :
		count = {'senior':169,'adult':3131,'child':312,'teen':344}
	if label_name == 'face' :
		count = {'smiling':1879,'serious':1977,'funny':100} 
	if label_name == 'race' :
		count = {'black':313,'white':3502,'other':21,'hispanic':39,'asian':81} 
	
	if balance :
		count_lst = list(count.values())
		count_sum = sum(count_lst)
		percent = [x/count_sum for x in count_lst]
		duplicate_ratio = []
		a = 1/len(percent)
		for i in range(len(percent)) :
			if percent[i] < a :
				b = int((a/percent[i])/1.5)
				duplicate_ratio.append(b)
			else :
				duplicate_ratio.append(0)
		duplicate_ratio = dict(zip(count.keys(),duplicate_ratio))


	for key,value in count.items() :
		count[key] = int(value*ratio)
	

	for i in x:
		l = label[i][label_name]
		data = img_data[i]
		if count[l] > 0 :
			train_data = np.vstack((train_data, data))
			train_label.append(label[i])
			if balance :
				if duplicate_ratio[l] != 0 :
					B = np.tile(data, (duplicate_ratio[l], 1))
					duplicate_data = np.vstack((duplicate_data, B))
					for j in range(duplicate_ratio[l]):
						duplicate_label.append(label[i])

			count[l] -= 1
		else :
			test_data = np.vstack((test_data, data))
			test_label.append(label[i])
	
	train_data = np.delete(train_data,0,0)
	test_data = np.delete(test_data,0,0)
	if balance :
		duplicate_data = np.delete(duplicate_data,0,0)
		train_data = np.vstack((train_data,duplicate_data))

	duplicate_len = len(duplicate_label)
	train_len = len(train_label)
	test_len = len(test_label)

	full_label = train_label + duplicate_label + test_label
	full_label,_ = encode(full_label,label_name)

	
	train_label = full_label[ : train_len+duplicate_len]
	test_label = full_label[train_len+duplicate_len : ]
	

	train_label = np.array(train_label)
	test_label = np.array(test_label)
	
	
	print('划分完成，训练集/测试集=',ratio)
	if balance :
		print('训练集过采样数量为：',duplicate_len)
	return train_data,train_label,test_data,test_label