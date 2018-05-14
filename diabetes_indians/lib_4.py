import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def norm_arr_std(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


def norm_minmax(df):
    return (df - df.min())/(df.max() - df.min())


def norm_df(df):
    result = df.copy()

    for feature in df.columns:
        result[feature] = norm_arr_std(result[feature])

    return result


def split(df, perc):
    msk = np.random.rand(len(df['mass'])) < perc
    return df[msk], df[~msk]


def printMeanStd(dframe):
    for feature in dframe.columns:
        print(dframe[feature].mean(), dframe[feature].std())


def plot(df, df_std):
    plt.figure(figsize=(8, 6))

    plt.scatter(df['preg'], df['mass'],
                color='green', label='input scale', alpha=0.5)

    plt.scatter(df_std['preg'], df_std['mass'], color='red',
                label='standardized', alpha=0.3)

    plt.title('Plasma and Insulin values of the diabetes dataset')
    plt.xlabel('Pregnancy No.')
    plt.ylabel('Body Mass')
    plt.legend(loc='upper left')
    plt.grid()

    plt.tight_layout()

    plt.show()


def plot_class(df, df_std):
    fig, ax = plt.subplots(2, figsize=(6, 14))

    for a, d, l in zip(range(len(ax)),
                       (df[['plas', 'test']].values,
                        df_std[['plas', 'test']].values),
                       ('Input scale', 'Standardized')):
        for i, c in zip(range(0, 2), ('red', 'green')):
            ax[a].scatter(d[df['class'].values == i, 0],
                          d[df['class'].values == i, 1],
                          alpha=0.5,
                          color=c,
                          label='Class %s' % i
                          )
        ax[a].set_title(l)
        ax[a].set_xlabel('Plasma')
        ax[a].set_ylabel('Insulin')
        ax[a].legend(loc='upper left')
        ax[a].grid()

    plt.tight_layout()
    plt.show()


def load_diabetes_df():
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    return pd.read_csv('./data.csv', names=names)


def split_train_test(df, perc):
    splitted_1 = split(df[df['class'] == 1], perc)
    splitted_0 = split(df[df['class'] == 0], perc)
    return [pd.concat([splitted_0[0], splitted_1[0]]), pd.concat([splitted_0[1], splitted_1[1]])]

