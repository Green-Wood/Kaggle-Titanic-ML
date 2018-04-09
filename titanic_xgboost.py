from sklearn.cross_validation import train_test_split
from sklearn import metrics
from data_process import data_clean
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV


def learning_rate():
    score_min = []
    test_time = 8
    for i in range(test_time):        # 测试8次
        train_X, test_X, train_Y, test_Y = train_test_split(source_X, source_y, train_size=.68)

        model = XGBClassifier(learning_rate=0.01, max_depth=4, n_estimators=500, silent=True, objective='binary:logistic')

        model.fit(train_X, train_Y)
        score = metrics.accuracy_score(test_Y, model.predict(test_X))
        score_min.append(score)
        print('{}:'.format(i + 1), score)

    print('the min is :', min(score_min))


def n_estimator():
    train_X, test_X, train_Y, test_Y = train_test_split(source_X, source_y, train_size=.68)
    model = XGBClassifier(learning_rate=0.01, silent=True, objective='binary:logistic', max_depth=4)
    param_test = {'n_estimators': list(range(300, 500, 5))}
    grid_search = GridSearchCV(estimator=model, param_grid=param_test, scoring='accuracy', cv=5)
    grid_search.fit(train_X, train_Y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)


def max_depth():
    best_para = []
    best_score = []
    for i in range(10):
        train_X, test_X, train_Y, test_Y = train_test_split(source_X, source_y, train_size=.68)
        model = XGBClassifier(learning_rate=0.01, silent=True, objective='binary:logistic', n_estimators=350)
        param_test = {'max_depth': list(range(1, 8, 1))}
        grid_search = GridSearchCV(estimator=model, param_grid=param_test, scoring='accuracy', cv=5)
        grid_search.fit(train_X, train_Y)
        best_para.append(grid_search.best_params_)
        best_score.append(grid_search.best_score_)

    for i in range(10):
        print(best_para[i])
        print(best_score[i])


full_X, full = data_clean()

sourceRow = 891         # 训练数据集的大小
# 原始数据集：特征
source_X = full_X.loc[0:sourceRow-1, :]
# 原始数据集：标签
source_y = full.loc[0:sourceRow-1, 'Survived']

max_depth()