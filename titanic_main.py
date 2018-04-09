import pandas as pd
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
from data_process import data_clean

full_X, full = data_clean()

sourceRow = 891

source_X = full_X.loc[0:sourceRow-1, :]

source_y = full.loc[0:sourceRow-1, 'Survived']


pred_X = full_X.loc[sourceRow:, :]

train_X, test_X, train_Y, test_Y = train_test_split(source_X, source_y, train_size=.8)

model = XGBClassifier(learning_rate=0.22, max_depth=4, n_estimators=25, silent=True, objective='binary:logistic')
model.fit(source_X, source_y)

pred_Y = model.predict(pred_X)
pred_Y = pred_Y.astype(int)

passenger_id = full.loc[sourceRow:, 'PassengerId']

predDf = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred_Y})
predDf.to_csv('34-Titanic_pred(lr = 0.22), age-mean.csv', index=False)

# 最好成绩(0.81339)：XGBClassifier(learning_rate=0.22, max_depth=4, n_estimators=25, silent=True, objective='binary:logistic')
# age-mean
