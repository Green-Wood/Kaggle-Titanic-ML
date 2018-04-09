from data_process import data_clean
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier
, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from xgboost import XGBClassifier


def train_svm(kfold, source_X, source_y):
    # SVM Training
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel': ['rbf'],
                      'gamma': [0.001, 0.01, 0.1, 1],
                      'C': [1, 10, 50, 100, 200, 300, 1000]}

    gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", verbose=1)
    gsSVMC.fit(source_X, source_y)
    SVMC_best = gsSVMC.best_estimator_

    # Best score
    print('gsSVMC:', gsSVMC.best_score_, SVMC_best)
    return SVMC_best


def best_svm():
    SVMC = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)
    return SVMC


def train_ada(kfold, source_X, source_y):
    # ada Training
    DTC = DecisionTreeClassifier()
    adaDTC = AdaBoostClassifier(DTC, random_state=7)

    ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                      "base_estimator__splitter": ["best", "random"],
                      "algorithm": ["SAMME", "SAMME.R"],
                      "n_estimators": [1, 2],
                      "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

    gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", verbose=1)
    gsadaDTC.fit(source_X, source_y)
    ada_best = gsadaDTC.best_estimator_
    print('gsadaDTC', gsadaDTC.best_score_, ada_best)
    return ada_best


def best_ada():
    adaDTC = AdaBoostClassifier(algorithm='SAMME.R',
                                base_estimator=DecisionTreeClassifier(class_weight=None, criterion='entropy',
                                                                      max_depth=None,
                                                                      max_features=None, max_leaf_nodes=None,
                                                                      min_impurity_decrease=0.0,
                                                                      min_impurity_split=None,
                                                                      min_samples_leaf=1, min_samples_split=2,
                                                                      min_weight_fraction_leaf=0.0, presort=False,
                                                                      random_state=None,
                                                                      splitter='best'),
                                learning_rate=0.01, n_estimators=2, random_state=7)
    return adaDTC


def train_extra_trees(kfold, source_X, source_y):
    # ExtraTrees
    ExtC = ExtraTreesClassifier()
    # Search grid for optimal parameters
    ex_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}

    gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", verbose=1)
    gsExtC.fit(source_X, source_y)
    ExtC_best = gsExtC.best_estimator_
    print('gsExtc:', gsExtC.best_score_, ExtC_best)
    return ExtC_best


def best_extra_trees():
    Extc = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                                max_depth=None, max_features=1, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=10, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                                oob_score=False, random_state=None, verbose=0, warm_start=False)
    return Extc


def train_random_forest(kfold, source_X, source_y):
    # RFC Parameters tunning
    RFC = RandomForestClassifier()
    # Search grid for optimal parameters
    rf_param_grid = {"max_depth": [None],
                     "max_features": [1, 3, 10],
                     "min_samples_split": [2, 3, 10],
                     "min_samples_leaf": [1, 3, 10],
                     "bootstrap": [False],
                     "n_estimators": [100, 300],
                     "criterion": ["gini"]}

    gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", verbose=1)
    gsRFC.fit(source_X, source_y)
    RFC_best = gsRFC.best_estimator_
    print('gsRFC:', gsRFC.best_score_, RFC_best)
    return RFC_best


def best_random_forest():
    RFC = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                                 max_depth=None, max_features=3, max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=3, min_samples_split=10,
                                 min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
                                 oob_score=False, random_state=None, verbose=0,
                                 warm_start=False)
    return RFC


def train_gradient_boosting(kfold, source_X, source_y):
    # gradient boosting
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss': ["deviance"],
                     'n_estimators': [100, 200, 300],
                     'learning_rate': [0.1, 0.05, 0.01],
                     'max_depth': [4, 8],
                     'min_samples_leaf': [100, 150],
                     'max_features': [0.3, 0.1]
                     }

    gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", verbose=1)

    gsGBC.fit(source_X, source_y)
    GBC_best = gsGBC.best_estimator_
    print('gsGBC:', gsGBC.best_score_, GBC_best)
    return GBC_best


def best_gradient_boosting():
    GBC = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                     learning_rate=0.1, loss='deviance', max_depth=4,
                                     max_features=0.3, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=100, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=300,
                                     presort='auto', random_state=None, subsample=1.0, verbose=0,
                                     warm_start=False)
    return GBC


def train_xgb_boosting(kfold, source_X, source_y):
    xgb = XGBClassifier(silent=True, objective='binary:logistic', learning_rate=0.22)
    xgb_param_grid = {
        'n_estimators': list(range(15, 30, 1)),
        'max_depth': [2, 4, 5, 6, 7]
    }
    gsXGB = GridSearchCV(xgb, param_grid=xgb_param_grid, cv=kfold, scoring='accuracy', verbose=1)
    gsXGB.fit(source_X, source_y)
    XGB_best = gsXGB.best_estimator_
    print('gsXGB:', gsXGB.best_score_, XGB_best)
    return XGB_best


def best_xgb_boosting():
    model = XGBClassifier(learning_rate=0.22, max_depth=4, n_estimators=25, silent=True, objective='binary:logistic')
    return model


full_X, full = data_clean()

sourceRow = 891  # 训练数据集的大小
# 原始数据集：特征
source_X = full_X.loc[0:sourceRow - 1, :]
# 原始数据集：标签
source_y = full.loc[0:sourceRow - 1, 'Survived']
# 预测数据集：特征
pred_X = full_X.loc[sourceRow:, :]

kfold = StratifiedKFold(n_splits=10)

# SVMC_best = best_svm()            #best 函数是在当前最好参数下生成的
# ada_best = best_ada()
# ExtC_best = best_extra_trees()
# RFC_best = best_random_forest()
# GBC_best = best_gradient_boosting()

# SVMC_best = train_svm(kfold, source_X, source_y)
# ada_best = train_ada(kfold, source_X, source_y)
# ExtC_best = train_extra_trees(kfold, source_X, source_y)
# RFC_best = train_random_forest(kfold, source_X, source_y)
# GBC_best = train_gradient_boosting(kfold, source_X, source_y)
# XGB_best = best_xgb_boosting(kfold, source_X, source_y)

print('Training is complete')

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
                                       ('svc', SVMC_best), ('gbc', GBC_best), ('ada', ada_best)], voting='soft')

votingC.fit(source_X, source_y)

pred_Y = votingC.predict(pred_X)
pred_Y = pred_Y.astype(int)

passenger_id = full.loc[sourceRow:, 'PassengerId']

predDf = pd.DataFrame({'PassengerId': passenger_id, 'Survived': pred_Y})
predDf.to_csv('49-Titanic_pred(best params func mean age).csv', index=False)
