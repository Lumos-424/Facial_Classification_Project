import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from util import Get_label, encode

from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

'''
采用经典集成学习方法进行该人脸识别数据集的单标签分类任务
包括bagging、随机森林、Adaboost、Stacking算法
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
age_label,_ = encode(labe_dict,'age')
race_label,_ = encode(labe_dict,'race')
face_label,_ = encode(labe_dict,'face')
prop_label,_ = encode(labe_dict,'prop')

# 选择需要进行分类的标签
label_seleted = sex_label

count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
num_class = 0
for i in label_seleted:
	for j in range(15):
		if i == j:
			count[i]+=1
print('该标签下各类别数据分布：',count)
for i in range(15):
    if count[i]!= 0:
        num_class+=1
print('该标签下是 %d 分类问题'%num_class)
# 归一化处理
scaler = StandardScaler()
pca_data_std = scaler.fit_transform(pca_data)

# elasticnet特征选择
eCV_model = ElasticNetCV(cv=10)
eCV_model.fit(pca_data_std, label_seleted)

# 取特征选择后系数不为零的特征
selected_features = eCV_model.coef_ != 0
X_selected = pca_data_std[:, selected_features]
print("特征选择后数据大小：",X_selected.shape)

#使用knn、贝叶斯、svm、决策树进行性别分类
clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5)
clf_tree = tree.DecisionTreeClassifier(criterion="entropy")
clf_svc = SVC(kernel='rbf', probability=True)
clf_gnb = GaussianNB()
models = [('KNN',clf_knn), ('NB',clf_gnb), ('SVM',clf_svc), ('DT',clf_tree)]
score = []
# for model in models :
# 	s = cross_val_score(model[1], X_selected, label_seleted, cv=5)
# 	print("Accuracy of %s: %0.2f (+/- %0.2f)" % (model[0],s.mean(), s.std() * 2))

# 使用简单BP神经网络进行分类
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X_selected, label_seleted, test_size=0.3, random_state=42)
clf_BP = MLPClassifier(
	random_state=1, max_iter=200, alpha=0.001, activation='relu',hidden_layer_sizes=(300,150))
# param_find = {
# 	'hidden_layer_sizes':[(10,), (20,), (30,)],
# 	'activation':['identity','logistic', 'tanh', 'relu'],
# 	'alpha':[0.1, 0.01, 0.001, 0.0001]
# }
#grid_search = GridSearchCV(clf_BP, param_grid=param_find, cv=5)
#grid_search.fit(X_train, y_train)
# 显示最优参数
#print('Best parameters of BP: ',grid_search.best_params_)

#clf_best_BP = grid_search.best_estimator_
#y_pred = clf_best_BP.predict(X_test)
clf_BP.fit(X_train, y_train)
y_pred = clf_BP.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy of easy BP:', acc)

#使用bagging、随机森林、adaboost、stacking集成学习方法进行性别分类
from sklearn.ensemble import BaggingClassifier
# bagging 算法只能采用bp神经网络、决策树、k近邻作为子分类器（方差大）
# bagging使用的子分类器(cv=5交叉验证评分)：DT-->0.71;KNN-->0.69；简单BP-->0.77
bagging = BaggingClassifier(clf_BP,n_estimators=30, max_samples=0.5, max_features=0.5)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)

from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(estimator=clf_svc, n_estimators=5) # 分类器串联训练，较慢

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
stacking = StackingClassifier(
	estimators=[('KNN',clf_knn), ('SVC',clf_svc), ('NB',clf_gnb), ('DT',clf_tree),('BP',clf_BP)],
	final_estimator=LogisticRegression()
)
model_assume = [("bagging",bagging),("rf",rf), ("adaboost",adaboost), ("stacking",stacking)] #adaboost跑得非常慢
score_assume = []
# for model in model_assume :
# 	s = cross_val_score(model[1], X_selected, sex_label, cv=5)
# 	print("Accuracy of %s: %0.2f (+/- %0.2f)" % (model[0], s.mean(), s.std() * 2))
print()
# 采用score方法
for model in model_assume:
	time_start = time.time()
	model[1].fit(X_train, y_train)
	y_pred = model[1].predict(X_test)
	time_end = time.time()
	time_cost = time_end - time_start
	accuracy = accuracy_score(y_test, y_pred)

	print('Accuracy of %s:%0.4f' % (model[0], accuracy))
	print('Cost time(fit + pred) of %s is %.2f s' % (model[0], time_cost))
	print()