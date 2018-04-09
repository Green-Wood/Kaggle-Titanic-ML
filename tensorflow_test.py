from sklearn.cross_validation import train_test_split
from sklearn import metrics
import tensorflow as tf
from data_process import data_clean


full_X, full = data_clean()

sourceRow = 891         # 训练数据集的大小
# 原始数据集：特征
source_X = full_X.loc[0:sourceRow-1, :]
# 原始数据集：标签
source_y = full.loc[0:sourceRow-1, 'Survived']

# 预测数据集：特征
pred_X = full_X.loc[sourceRow:, :]

train_X, test_X, train_Y, test_Y = train_test_split(source_X, source_y, train_size=.75)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=27)]

model = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], feature_columns=feature_columns)
model.fit(test_X, test_Y)
score = metrics.accuracy_score(test_Y, model.predict(test_X))
print(score)
