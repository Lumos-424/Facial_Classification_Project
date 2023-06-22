from torchvision import datasets,models,transforms
from torch.utils.data import Dataset,DataLoader
from util import Get_label, encode, DeleteBadData
from PIL import Image
import torch
import numpy as np
import random

#---------------------------------------------------------------------------
#----------------------------自定义数据集------------------------------------
class FaceDataset(Dataset):
	def __init__(self,label,img_data, transform=None):
		super().__init__()
		self.label = label             #包含所有标签信息的list         
		self.transfrom = transform            
		self.img_data = img_data

	def __len__(self) :    #返回数据集的大小
		return len(self.label)

	def __getitem__(self,index):
		
		img_data = self.img_data[index]
		img_data = img_data.reshape(128,128)
		img_data=Image.fromarray(img_data)             #将图片从numpy转化为PIL
		img_data = self.transfrom(img_data)           #转化为tensor并归一化

		label = torch.tensor(self.label[index])

		return img_data, label
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

