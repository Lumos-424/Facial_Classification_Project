import numpy as np
import cv2
import util
from util import Get_label,DeleteBadData,Get_img,img2jpg
import os

data_path = 'data/img.npy'   #保存原始图像npy文件路径
img_path = '../face/rawdata/' #原始图像路径
detected_path = 'detected/' #保存人脸检测jpg路径
label_Path = "../face/"    #标签路径
jpg_path = 'jpg/'         #保存原始图像jpg路径
cas_alt2 = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml") #读取haar级联器

label_dict = Get_label(label_Path)
label_all = []
for dict in label_dict :            #对于缺失数据直接删除
    if dict['missing'] != 'true' :
        label_all.append(dict)
    else :
        continue

if os.path.exists(data_path):
    img_data = np.load(data_path)
else :
    img_data = Get_img(img_path,label_all)

print('剔除缺失数据后的图片数量：',len(label_all))
img_data,label_all = DeleteBadData(img_data,label_all)
print('剔除错误数据后的图片数量：',len(label_all))
print('正在保存jpg图片....')
img2jpg(img_data,jpg_path,label_all)
print('jpg图片保存完成，保存路径为：',jpg_path)

print('正在进行人脸检测.......')

id = label_all[0]['id']
img = cv2.imread(jpg_path + id + '.jpg',cv2.IMREAD_GRAYSCALE)
face = cas_alt2.detectMultiScale(img)
for (x, y, w, h) in face:
	img=img[x:x+w,y:y+h]
img = cv2.resize(img,(128,128))
cv2.imwrite(detected_path+id+'.jpg',img)
img_data = img.reshape(1,16384)

empty = []
multi = []
for i in range(1,len(label_all)) :
	id = label_all[i]['id']
	img = cv2.imread(jpg_path + id + '.jpg',cv2.IMREAD_GRAYSCALE)
	face = cas_alt2.detectMultiScale(img)
	
	if type(face) == type(()):       #无检测框，直接用原图
		cv2.imwrite(detected_path+id+'.jpg',img)
		img = img.reshape(1,16384)
		img_data = np.vstack((img_data, img))
		empty.append(id)
		continue

	if face.shape[0] == 1 :    #检测到一个框可以直接裁剪出人脸再reshape回128*128使用
		for (x, y, w, h) in face:
			img=img[x:x+w,y:y+h]
		img = cv2.resize(img,(128,128))
		cv2.imwrite(detected_path+id+'.jpg',img)
		img = img.reshape(1,16384)
		img_data = np.vstack((img_data, img))
	else :                  #同一个人检测多个框用原图
		cv2.imwrite(detected_path+id+'.jpg',img)
		img = img.reshape(1,16384)
		img_data = np.vstack((img_data, img))
		multi.append(id)
		continue
print('人脸检测完成，检测失败图像id为：',empty)
print('一张图像多个检测框的id为：',multi)
print('人脸检测图像保存路径为：',detected_path)
print('人脸检测后的图像数组shape为：',img_data.shape)

np.save('data/detected_face.npy',img_data)



