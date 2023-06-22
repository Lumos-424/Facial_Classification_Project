文件：
`data.zip`：包含img.npy文件和pca_img.npy，即直接转化原始图像得到的像素点序列和经过pca提取后的特征序列

`add_FS.py`:包括PCA法特征提取、ElasticNetCV进行特征选择、以及分别采用四种经典集成学习算法运行人脸识别任务的内容

`easy_BP.py`:使用MLPCLassifier，通过网格搜寻查找最优参数

`GBDTx3.py`：三种使用梯度决策树原理的算法：GBDT、XGBoost、CatBoost

`param_search.py`:进行GBDT、XGBoost的最优参数搜寻

`util.py`:包含提取标签、标签编码等一系列预测处理函数

`img.npy`:将原始图像数据所有像素值的存储序列

`pca_img.npy`:通过PCA法提取特征后得到的数据序列

文件夹：（人脸图像识别）

`face->rawdata`:人脸识别数据集原始像素编码数据

`face->faceDR、faceDS`:数据集有数据对于标签


