import pandas as pd


def data_clean():                  # 处理清洗数据，并将full_X（数据集）、full（所有原始数据）返回
    train = pd.read_csv("./train.csv")  # train data
    test = pd.read_csv("./test.csv")  # test data

    full = train.append(test, ignore_index=True)  # 合并两个数据集以方便清洗和处理
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
    full['title'] = full['Name'].map(getTitle)  # map是函数式编程，可以将title_mapDict看成一个函数
    full['title'] = full['title'].map(title_mapDict)

    # pd.options.mode.chained_assignment = None
    # full['Age'][full['title'] == 'Master'] = full['Age'][full['title'] == 'Master'].fillna(4.0)
    # full['Age'][full['title'] == 'Miss'] = full['Age'][full['title'] == 'Miss'].fillna(22.0)
    # full['Age'][full['title'] == 'Mr'] = full['Age'][full['title'] == 'Mr'].fillna(30.0)
    # full['Age'][full['title'] == 'Mrs'] = full['Age'][full['title'] == 'Mrs'].fillna(35.0)
    # full['Age'][full['title'] == 'Others'] = full['Age'][full['title'] == 'Others'].fillna(39.5)
    # 依据不同头衔用中位值替换年龄空值
    full['Age'] = full['Age'].fillna(full['Age'].mean())
    full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
    full['Embarked'] = full['Embarked'].fillna('S')  # 用出现最频繁的值代替空值
    full['Cabin'] = full['Cabin'].fillna('U')  # 船舱号缺失值较多，因此用0表示没有船舱号，1表示有船舱号
    full['Cabin'] = full['Cabin'].map(lambda s: 0 if s == 'U' else 1)

    sex_mapDict = {'male': 1, 'female': 0}
    full['Sex'] = full['Sex'].map(sex_mapDict)  # 将性别按照字典定义替换为1和0

    embarkedDf = pd.DataFrame()
    embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')

    pclassDf = pd.DataFrame()
    pclassDf = pd.get_dummies(full['Pclass'], prefix='Pclass')

    titleDf = pd.DataFrame()
    titleDf = pd.get_dummies(full['title'])

    # cabinDf = pd.DataFrame()
    # full['Cabin'] = full['Cabin'].map(lambda c: c[0])
    # cabinDf = pd.get_dummies(full['Cabin'], prefix='Cabin')

    familyDf = pd.DataFrame()
    familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1
    familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    full_X = pd.concat([titleDf, pclassDf,
                        familyDf, full['Fare'], embarkedDf, full['Sex'], full['Cabin']], axis=1)
    full_X['isChild'] = full['Age'].map(lambda s: 1 if 0 <= s <= 15 else 0)

    return full_X, full


def getTitle(name):       # 从姓名中提取头衔
    str1 = name.split(',')[1]
    str2 = str1.split('.')[0]
    str3 = str2.strip()
    return str3


if __name__ == '__main__':
    train = pd.read_csv("./train.csv")  # train data
    test = pd.read_csv("./test.csv")  # test data

    full = train.append(test, ignore_index=True)  # 合并两个数据集以方便清洗和处理

    full["Age"] = full["Age"].fillna(full['Age'].median())  # 用平均值替换空值
    full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
    full['Embarked'] = full['Embarked'].fillna('S')  # 用出现最频繁的值代替空值
    full['Cabin'] = full['Cabin'].fillna('U')  # 船舱号缺失值较多，因此用U——Unknown来表示空值
    full['Cabin'] = full['Cabin'].map(lambda s: 0 if s == 'U' else 1)

    sex_mapDict = {'male': 1, 'female': 0}
    full['Sex'] = full['Sex'].map(sex_mapDict)  # 将性别按照字典定义替换为1和0

    embarkedDf = pd.DataFrame()
    embarkedDf = pd.get_dummies(full['Embarked'], prefix='Embarked')

    pclassDf = pd.DataFrame()
    pclassDf = pd.get_dummies(full['Pclass'], prefix='Pclass')

    titleDf = pd.DataFrame()
    titleDf['Title'] = full['Name'].map(getTitle)  # map是函数式编程，可以将title_mapDict看成一个函数
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
    titleDf['Title'] = titleDf['Title'].map(title_mapDict)
    titleDf = pd.get_dummies(titleDf['Title'])

    # cabinDf = pd.DataFrame()
    # full['Cabin'] = full['Cabin'].map(lambda c: c[0])
    # cabinDf = pd.get_dummies(full['Cabin'], prefix='Cabin')

    familyDf = pd.DataFrame()
    familyDf['FamilySize'] = full['Parch'] + full['SibSp'] + 1
    familyDf['Family_Single'] = familyDf['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    familyDf['Family_Small'] = familyDf['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    familyDf['Family_Large'] = familyDf['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    # corrDf = full.corr()    # 关系矩阵

    full_X = pd.concat([titleDf, pclassDf, familyDf, full['Fare'], embarkedDf, full['Sex'], full['Cabin']], axis=1)
    full_X['isChild'] = full['Age'].map(lambda s: 1 if 0 <= s <= 15 else 0)
    full_add_Survived = pd.concat([titleDf, pclassDf, familyDf, full['Fare'], full['Age'],
                                   embarkedDf, full['Sex'], full['Survived']], axis=1)

    print(full_X.head())
