import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets,models,transforms
import torchvision.models as models
from FaceDataset import FaceDataset
from ealyStop import EarlyStopping
from util import DeleteBadData, Get_label,train_test_split,Get_img
import os


# 定义train loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train() # 设置模型为训练模式
    size = len(dataloader.dataset)
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X,y = X.cuda(),y.cuda()
        # Compute prediction and loss
        pred = model(X)

        loss = loss_fn(pred, y)
    
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#        if batch % 100 == 0:
#            loss, current = loss.item(), (batch + 1) * len(X)
#            print(f"Train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= batch
    print(f"Train info: \n Avg loss: {train_loss:>8f} \n")


# 定义test loop
def test_loop(dataloader, model, loss_fn):
    model.eval() # 设置模型为评估/测试模式
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.cuda(),y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test info: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


#-----------------------超参数设置-----------------------------------
label_Path = "../face/"
data_path = 'data/img.npy'
img_path = '../face/rawdata/'
learning_rate = 1e-3
batch_size = 64
epochs = 100
label_name = 'race'         #对性别进行分类
save_path = "data/model.pth" #当前目录保存模型文件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#--------------------------------------------------------------------
label_dict = Get_label(label_Path)
label = []
for dict in label_dict :            #对于缺失数据直接删除
    if dict['missing'] != 'true' :
        label.append(dict)
    else :
        continue

if os.path.exists(data_path):
    img_data = np.load(data_path)
else :
    print('正在读取图像文件..')
    img_data = Get_img(img_path,label)


img_data,label = DeleteBadData(img_data,label)

transform = transforms.Compose([
                transforms.Resize((128,128)),   
                transforms.ToTensor()           #转化为tensor并归一化
])

train_data,train_label,test_data,test_label = train_test_split(img_data,label,label_name,ratio = 0.8,balance=True)
train_label = train_label.tolist()
test_label = test_label.tolist()

cls_num = len(np.bincount(train_label) )
train_dataset = FaceDataset(train_label,train_data,transform=transform)
test_dataset = FaceDataset(test_label,test_data,transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

resnet_18 = models.resnet18(pretrained=True)    #使用resnet18网络结构
resnet_18.fc = torch.nn.Linear(512, cls_num)           #修改全连接层的分类数目
resnet_18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #修改卷积核，因为图片单通道
model = resnet_18.to(device)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(save_path)
for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	train_loop(train_loader, model, loss_fn, optimizer)
	val_loss = test_loop(test_loader, model, loss_fn)
	early_stopping(val_loss, model)
    #达到早停止条件时，early_stop会被置为True
	if early_stopping.early_stop:
		print("Early stopping")
		break #跳出迭代，结束训练
print("Done!")
