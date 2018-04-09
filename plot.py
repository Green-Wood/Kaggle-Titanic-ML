from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from data_process import data_clean
import Ensembling


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == '__main__':
    full_X, full = data_clean()

    sourceRow = 891  # 训练数据集的大小
    # 原始数据集：特征
    source_X = full_X.loc[0:sourceRow - 1, :]
    # 原始数据集：标签
    source_y = full.loc[0:sourceRow - 1, 'Survived']
    # 预测数据集：特征
    pred_X = full_X.loc[sourceRow:, :]

    kfold = StratifiedKFold(n_splits=10)

    g = plot_learning_curve(Ensembling.best_xgb_boosting(),
                            "best XGB boosting curves(v1.0)", source_X, source_y, cv=kfold)
    g.draw()
    g.show()