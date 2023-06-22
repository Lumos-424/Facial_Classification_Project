import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,models,transforms
import torchvision.models as models
from FaceDataset import FaceDataset
from ealyStop import EarlyStopping
from util import DeleteBadData, Get_label,train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import torch.nn.functional as F




#-----------------------超参数设置-----------------------------------
label_Path = "../face/"
img_data = np.load('data/img.npy')
detected_face = np.load('data/detected_face.npy')
learning_rate = 1e-4
batch_size = 64
epochs = 100
label_name = 'age'         #对性别进行分类
save_path = "data/model.pth" #当前目录保存模型文件

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#--------------------------------------------------------------------

transform = transforms.Compose([
                transforms.Resize((128,128)),   
                transforms.ToTensor()           #转化为tensor并归一化
])



test_data = np.load('data/test_data.npy')
test_label = np.load('data/test_label.npy')
test_label = test_label.tolist()

cls_num = len(np.bincount(test_label) )

test_dataset = FaceDataset(test_label,test_data,transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(pretrained=False)    #使用resnet18网络结构
model.fc = torch.nn.Linear(512, cls_num)           #修改全连接层的分类数目
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #修改卷积核，因为图片单通道
model.load_state_dict(torch.load(save_path))
model = model.to(device)
model.eval()

predict = []
with torch.no_grad():
    for X, y in test_loader:
        X,y = X.cuda(),y.cuda()
        pred = model(X)
        pred = pred.argmax(1)
        predict  = predict + pred.tolist()
print(predict)
print(test_label.count(4))
predict = torch.tensor(predict)
predict = F.one_hot(predict)
test_label = torch.tensor(test_label)
test_label = F.one_hot(test_label)


fp_mean = 0
tp_mean = 0
aucs = []
#curve_name = ['senior','adult','child','teen']
#curve_name = ['smiling','serious','funny']
curve_name = ['black','white','other','hispanic','asian']
# curve_name = ['female','male']
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.title('人脸检测后性别分类ROC曲线')

# fpr, tpr, thresholds = roc_curve(test_label, predict)
# plt.plot(fpr, tpr, label='性别(area = {:.3f})'.format(auc(fpr, tpr)))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.legend()
for i in range(predict.shape[1]) :
    fpr, tpr, thresholds = roc_curve(test_label[:,i], predict[:,i])
    fp_mean = fpr[1] + fp_mean
    tp_mean = tpr[1] + tp_mean
    plt.plot(fpr, tpr, label=curve_name[i] + '(area = {:.3f})'.format(auc(fpr, tpr)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    aucs.append(auc(fpr, tpr))
fp_mean = fp_mean/predict.shape[1]
tp_mean = tp_mean/predict.shape[1]
fp_mean = np.array([0,fp_mean,1])
tp_mean = np.array([0,tp_mean,1])
plt.plot(fp_mean, tp_mean, label='adverage(area = {:.3f})'.format(np.mean(aucs)),linestyle = 'dashdot')

plt.legend()
plt.show()
