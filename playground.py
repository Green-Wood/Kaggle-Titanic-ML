from data_process import data_clean
from sklearn.cross_validation import train_test_split
import Ensembling

full_X, full = data_clean()

sourceRow = 891  # 训练数据集的大小
# 原始数据集：特征
source_X = full_X.loc[0:sourceRow - 1, :]
# 原始数据集：标签
source_y = full.loc[0:sourceRow - 1, 'Survived']

train_X, test_X, train_Y, test_Y = train_test_split(source_X, source_y, train_size=.68)