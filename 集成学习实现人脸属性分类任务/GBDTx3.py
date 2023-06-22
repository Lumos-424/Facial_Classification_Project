import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import catboost as cb
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA # 特征提取

from util import Get_label, encode, DeleteBadData
import time
'''
包括三种基于梯度提升决策树的集成学习算法
GBDT、XGBoost、CatBoost
'''
# 数据导入
label_Path = "./人脸图像识别/face/"
label_all = Get_label(label_Path)
pca_data = np.load("pca_img.npy")

# 标签集提取
labe_dict = []
for dict in label_all:  # 对于标签缺失的数据直接删除
    if dict['missing'] != 'true':
        labe_dict.append(dict)  # labe_dict就是直接去除缺失数据后的label.txt的内容
    else:
        continue

sex_label, _ = encode(labe_dict, 'sex')
age_label,_ = encode(labe_dict,'age')
race_label,_ = encode(labe_dict,'race')
face_label,_ = encode(labe_dict,'face')
prop_label,_ = encode(labe_dict,'prop')

# 选择需要分类的标签
label_seleted = sex_label
# 统计类别数目
# prop标签有15类，所以设置最多为15个统计位
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
print('pca提取后的数据大小:', pca_data_std.shape)

# elasticnet特征选择
eCV_model = ElasticNetCV(cv=10)
eCV_model.fit(pca_data_std, label_seleted)

# 取特征选择后系数不为零的特征
selected_features = eCV_model.coef_ != 0
X_selected = pca_data_std[:, selected_features]
print("特征选择后数据大小：", X_selected.shape)

# 统一划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, label_seleted, test_size=0.3, random_state=42)

# GBDT
clf_gbc = GradientBoostingClassifier(
    loss='log_loss',  # 交叉熵损失函数，适用于二分类问题
    learning_rate=0.1,
    n_estimators=400,
    max_depth=5,
    min_samples_split=2,
    verbose=0)

# XGBoost
# 定义模型参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': num_class,  # 类别数目
    'max_depth': 8, # 最大深度，合理设置能防止过拟合,key1
    'min_child_weight': 2,  # 叶子节点最小样本数,key2
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'n_estimators': 160,
    'reg_alpha':0,
    'reg_lambda':1,
    'verbosity':0
}
# 将数据转换为 DMatrix 格式(XGBoost特有数据格式，可加速训练）
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)
clf_xgb = xgb.XGBClassifier(**params)

# CatBoost
'''
best score" 0.8051297109308276
best params: {'depth': 7, 'iterations': 500, 'l2_leaf_reg': 9, 'learning_rate': 0.03}
'''
X_train, X_test, y_train, y_test = train_test_split(X_selected, label_seleted, test_size=0.2, random_state=0)
clf_cb = CatBoostClassifier(depth=7, iterations=500, l2_leaf_reg=9, learning_rate=0.03,verbose=0)


model_assume = [("GBDT",clf_gbc),("XGBoost",clf_xgb), ("CatBoost",clf_cb)]
# # 采用交叉验证的方法
# score_assume = []
# for model in model_assume :
# 	s = cross_val_score(model[1], X_selected, label_seleted, cv=5)
# 	print("Accuracy of %s: %0.3f (+/- %0.2f)" % (model[0], s.mean(), s.std() * 2))

#采用score方法
for model in model_assume:
	time_start = time.time()
	model[1].fit(X_train, y_train)
	y_pred = model[1].predict(X_test)
	time_end = time.time()
	time_cost = time_end - time_start
	accuracy = accuracy_score(y_test, y_pred)

	print('Accuracy of %s:%0.4f' % (model[0], accuracy))
	print('Cost time(fit + pred) of %s is %.2fs' % (model[0], time_cost))
	print()

