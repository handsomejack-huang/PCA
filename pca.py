import numpy as np
from sklearn.decomposition import PCA

X = np.random.randn(10, 5)
X -= X.mean(0)
n = 1
# sklearn 方法
pca = PCA(n_components=n)
new = pca.fit_transform(X)

# 特征值方法
# 求解协方差矩阵，因为矩阵已减去均值，可以直接使用矩阵乘法求解
X_cov = (1 / (len(X) - 1)) * np.dot(X.T,X)
# 求解协方差的特征值
eigVal, eigVect = np.linalg.eig(X_cov)
# 将特征值从大到小排序，选取n个特征值
eigInd = np.argsort(-eigVal)[:n]
X_n = np.dot(X, eigVect[:, eigInd])

print('sklearn与特征值法对比', abs(new) - abs(X_n))

# 奇异值方法,其中V是X.T * X 的特征向量，协方差矩阵为 （1/（10-1）*X.T * X）
U, S, V = np.linalg.svd(X)
# X_new = U * S * V.T * V = U * S,原矩阵乘V即为转化后的矩阵，取前n个向量即可
X_new = np.dot(U[:, :n], S[:n]).reshape(10, 1)
print('sklearn与奇异值对比', abs(new) - abs(X_new))
print(pca.explained_variance_ratio_)
print(eigVal[eigInd] / sum(eigVal))
ss = S ** 2
print(ss[:n] / sum(ss))
