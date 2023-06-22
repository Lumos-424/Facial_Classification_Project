from util import Get_label, encode
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
'''
负责简单神经网络的参数搜寻
'''
# 数据导入
label_Path = "./人脸图像识别/face/"
label_all = Get_label(label_Path)

img_data = np.load("img.npy")  #读取图像文件，详见readme.txt
pca_data = np.load("pca_img.npy")

# 标签集提取
labe_dict = []
for dict in label_all :            #对于标签缺失的数据直接删除
	if dict['missing'] != 'true' :
		labe_dict.append(dict) # labe_dict就是直接去除缺失数据后的label.txt的内容
	else :
		continue

sex_label,_ = encode(labe_dict,'sex')

# 归一化处理
scaler = StandardScaler()
pca_data_std = scaler.fit_transform(pca_data)

# elasticnet特征选择
eCV_model = ElasticNetCV(cv=10)
eCV_model.fit(pca_data_std, sex_label)

# 取特征选择后系数不为零的特征
selected_features = eCV_model.coef_ != 0
X_selected = pca_data_std[:, selected_features]
print("特征选择后数据大小：",X_selected.shape)

# 使用简单BP神经网络进行分类
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X_selected, sex_label, test_size=0.3, random_state=42)
clf_BP = MLPClassifier(random_state=1)
param_find = {
	'hidden_layer_sizes':[ (300,150,) , (400,200,) , (500, 250,),(300,150,50,), (400,200,100,) ],
	'activation':['relu'],
	'alpha':[0.01, 0.001, 0.0001],
    'max_iter':[200, 300, 400, 500, 600]
}
grid_search = GridSearchCV(clf_BP, param_grid=param_find, cv=5)
grid_search.fit(X_train, y_train)
# 显示最优参数
print('Best parameters of BP: ',grid_search.best_params_)

clf_best_BP = grid_search.best_estimator_
y_pred = clf_best_BP.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy of easy BP:', acc)