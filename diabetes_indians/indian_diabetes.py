import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


def norm_df(dframe):
    result = dframe.copy()

    for feature in df_init.columns:
        result[feature] = norm_arr(result[feature])

    return result


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
    fig, ax = plt.subplots(2, figsize=(6,14))

    for a,d,l in zip(range(len(ax)),
               (df[['plas', 'test']].values, df_std[['plas', 'test']].values),
               ('Input scale',
                'Standardized')
                ):
        for i,c in zip(range(0,2), ('red', 'green')):
            ax[a].scatter(d[df['class'].values == i, 0],
                  d[df['class'].values == i, 1],
                  alpha=0.5,
                  color=c,
                  label='Class %s' %i
                  )
        ax[a].set_title(l)
        ax[a].set_xlabel('Plasma')
        ax[a].set_ylabel('Insulin')
        ax[a].legend(loc='upper left')
        ax[a].grid()

    plt.tight_layout()
    plt.show()


names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df_init = pd.read_csv('./data.csv', names=names)

norm_preg = norm_arr(df_init['preg'])
print(norm_preg)

for feature in df_init.columns:
    nrm = norm_arr(df_init[feature])
    print(nrm.mean(), nrm.std())


df_norm = norm_df(df_init)
print(df_norm.head(10))

print()
printMeanStd(df_init)

print()
printMeanStd(df_norm)

# plot_class(df_init, df_norm)

print(df_init.loc('class'))

print(df_init['class'].mean())


msk = np.random.rand(len(df_init['mass'])) < 0.8

train = df_init[msk]
test = df_init[~msk]


def split(df, perc):
    msk = np.random.rand(len(df['mass'])) < perc

    train = df[msk]
    test = df[~msk]
    return train, test


splitted_1 = split(df_init[df_init['class'] == 1], 0.8)
splitted_0 = split(df_init[df_init['class'] == 0], 0.8)

print(splitted_1[0])

print(pd.concat(splitted_0[0], splitted_1[0]))
print(pd.concat(splitted_0[1], splitted_1[1]))