import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


def norm_df(df):
    result = df.copy()

    for feature in df.columns:
        result[feature] = norm_arr(result[feature])

    return result


def stratified_split(y, proportion = 0.8):
    y = np.array(y)

    train_inds = np.zeros(len(y), dtype=bool)
    test_inds = np.zeros(len(y), dtype=bool)

    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        np.random.shuffle(value_inds)

        n = int(proportion * len(value_inds))

        train_inds[value_inds[:n]] = True
        test_inds[value_inds[n:]] = True

    return train_inds, test_inds


def norm_df(dframe):
    result = dframe.copy()

    for feature in dframe.columns:
        result[feature] = norm_arr(result[feature])

    return result

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv('./data.csv', names=names)


def calculate_precision(real, pred):
    compare_df = pd.DataFrame({'real': real, 'predicted': pred})
    compare_df['correct'] = compare_df['real'] == compare_df['predicted']
    return compare_df['correct'].mean()


def calc_accuracy(real, pred):
    return 1 - sum((abs(real - pred))/len(real))


def calc_bcr_accuracy(real, pred):
    compare_df = pd.DataFrame({'real': real, 'predicted': pred})
    compare_df['correct'] = compare_df['real'] == compare_df['predicted']
    grouped = compare_df.groupby(compare_df.real)['correct'].mean()
    return grouped.mean()


def CV(df, key, classifier, nfold, normalize):
    acc = []

    for i in range(nfold):
        train, test = stratified_split(df[key])

        x_train = df.iloc[train, 0:8]
        x_test = df.iloc[test, 0:8]

        if normalize is True:
            x_train = norm_df(x_train)
            x_test = norm_df(x_test)

        y_train = df[key][train]
        y_test = df[key][test]

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        acc.append(calc_bcr_accuracy(y_test, y_pred))

    return np.mean(acc)


print('LogReg normalized', CV(df, 'class', LogisticRegression(), 100, True))
print('LogReg unnormalized', CV(df, 'class', LogisticRegression(), 100, False))
print('Forest normalized', CV(df, 'class', RandomForestClassifier(), 100, True))
print('Forest unnormalized', CV(df, 'class', RandomForestClassifier(), 100, False))
