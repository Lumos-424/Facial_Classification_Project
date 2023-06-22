import matplotlib.pyplot as plt
import util
import numpy as np
import numpy as np
import torch 
from util import DeleteBadData

label_path = '../face/'
label = util.Get_label(label_path)
img_data = np.load('data/img.npy')
labe_dict = []
for dict in label :            #对于标签缺失的数据直接删除
	if dict['missing'] != 'true' :
		labe_dict.append(dict)
	else :
		continue
img_data,labe_dict = DeleteBadData(img_data,labe_dict)

label_sex,_ = util.encode(labe_dict,'sex')
label_age,_ = util.encode(labe_dict,'age')
label_face, _ = util.encode(labe_dict,'face')
label_race, _ = util.encode(labe_dict,'race')
label_prop, _ = util.encode(labe_dict,'prop')
sex_num = {}
for i in label_sex:
    sex_num[i] = label_sex.count(i)
print('sex:',sex_num) 

age_num = {}
for i in label_age:
    age_num[i] = label_age.count(i)
print('age:',age_num) 

face_num = {}
for i in label_face:
    face_num[i] = label_face.count(i)
print('face:',face_num) 

race_num = {}
for i in label_race:
    race_num[i] = label_race.count(i)
print('race:',race_num) 

prop_num = {}
for i in label_prop:
    prop_num[i] = label_prop.count(i)
print(prop_num)


plt.subplot(2,2,1)
plt.bar(['female','male'],list(sex_num.values()),color = ['r','g','b','c','m'])
plt.subplot(2,2,2)
plt.bar(['senior','adult','child','teen'],list(age_num.values()),color = ['r','g','b','c','m'])
plt.subplot(2,2,3)
plt.bar(['smiling','serious','funny'],list(face_num.values()),color = ['r','g','b','c','m'])
plt.subplot(2,2,4)
plt.bar(['black','white','other','hispanic','asain'],list(race_num.values()),color = ['r','g','b','c','m'])

plt.show()
