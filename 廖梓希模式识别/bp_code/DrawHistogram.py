import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,models,transforms
import torchvision.models as models
from FaceDataset import FaceDataset
from ealyStop import EarlyStopping
import math
from PIL import Image
from util import Get_label,DeleteBadData,Get_img,img2jpg
import matplotlib.pyplot as plt


label_Path = "./img/face/"
img_data = np.load('./file/img.npy')
label_dict = Get_label(label_Path)
label_all = []

for dict in label_dict :            #对于缺失数据直接删除
    if dict['missing'] != 'true' :
        label_all.append(dict)
    else :
        continue


img_mean = np.mean(img_data,axis=1) #计算图片均值(一张图片所有像素值的均值)
x_mean = np.mean(img_mean) 
x_var = np.var(img_mean)


print('方差为:',x_var)

x = (img_mean - x_mean)/math.sqrt(x_var)     #图像均值标准化
print('标准化后的图像平均值序列：',img_mean)
print('标准化后的图像平均值序列：',x)

s=np.arange(-4.5,4.5,0.01)
def f(x):             #标准正态分布函数
    return (np.e)**(-x**2/2)/(2*np.pi)**0.5

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.plot(s,f(s),label='标准正态分布曲线')   #画出标准正态分布函数
plt.hist(x,bins=100,density=True,histtype='stepfilled',label='图像数据分布') #画出数据直方图
plt.legend(loc='upper left')
plt.show()

index = np.where( (x > 10) | (x < -1.8) )  #
index = index[0] #index是一个元组，[0]提取出数据
id_abnormal = []
for i in index :
	l = label_all[i]
	id = l['id']
	id_abnormal.append(id)
print('异常数据的id为:',id_abnormal)
