import numpy as np
import pandas as pd
from diabetes_indians.lib_4 import norm_df


def stratified_split(y, proportion=0.8):
    y = np.array(y)

    train = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(y), dtype=bool)

    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y == value)[0]
        np.random.shuffle(value_inds)

        n = int(proportion * len(value_inds))

        train[value_inds[:n]] = True
        test[value_inds[n:]] = True

    return train, test


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


def predict(x_train, y_train, x_test, y_test, classifier):
    classifier.fit(x_train, y_train)
    return classifier.predict(x_test)


def cross_validate(df, key, classifier, nfold, normalize=True, bcr=True):
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
        if bcr is True:
            acc.append(calc_bcr_accuracy(y_test, y_pred))
        else:
            acc.append(calc_accuracy(y_test, y_pred))

    return np.mean(acc)

