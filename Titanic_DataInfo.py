import numpy as np
import pandas as pd


def getTicket(Ticket):

    try:
        return int(Ticket)
    except ValueError:
        return getTicket(Ticket[1:])


def getTitle(name):       # 从姓名中提取头衔
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3


train = pd.read_csv("./train.csv")  # train data
test = pd.read_csv("./test.csv")   # test data

print("train shape:", train.shape, "test shape:", test.shape)

full = train.append(test, ignore_index=True)   # 合并两个数据集以方便清洗和处理

# print("shape of merged data:", full.shape)
# print(full.head())   # 查看变量信息(前几行的数据)

# print(full.describe())   # 查看描述统计信息
# print(full.info())
# print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# print()
# print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
# print()
# familyDf = pd.DataFrame()
# train['FamilySize'] = full['Parch'] + full['SibSp'] + 1
# print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
# for feature in full:
#     print(feature)
full['title'] = full['Name'].map(getTitle)
title_mapDict = {
        'Capt': 'Mr',  # 'Officer'
        'Col': 'Mr',  # 'Officer'
        'Major': 'Mr',  # 'Officer'
        'Jonkheer': 'Others',
        'Don': 'Others',
        'Sir': 'Others',
        'Dr': 'Mr',  # 'Officer'
        'Rev': 'Mr',  # 'Officer'
        'the Countess': 'Others',
        'Dona': 'Others',
        'Mme': 'Mrs',
        'Mlle': 'Miss',
        'Ms': 'Mrs',
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Miss': 'Miss',
        'Master': 'Master',
        'Lady': 'Others'
    }
full = full.groupby(['Ticket'])
print(full.head())