import pandas as pd

df = pd.read_csv('./auto_data.csv')
df.columns = [
    'symboling',
    'normalized-losses',
    'make',
    'fuel-type',
    'aspiration ',
    'num-of-doors',
    'body-style',
    'drive-wheels',
    'engine-location',
    'wheel-base',
    'length',
    'width',
    'height',
    'curb-weight',
    'engine-type',
    'num-of-cylinders',
    'engine-size',
    'fuel-system',
    'bore',
    'stroke',
    'compression-ratio',
    'horsepower',
    'peak-rpm',
    'city-mpg',
    'highway-mpg',
    'price']

df['num-of-doors'].replace(['one', 'two', 'three', 'four', 'five'], [1, 2, 3, 4, 5], inplace=True)
mean = df[df['num-of-doors'] != '?']['num-of-doors'].mean()

df.loc[df['num-of-doors'] == '?'] = 0
print(df['num-of-doors'].mean())

df.loc[df['num-of-doors'] == 0] = mean
print(df['num-of-doors'].mean())

numeric_prices = pd.to_numeric(df['price'], errors='coerce')
df['price'] = numeric_prices
df.loc[df['price'].isnull()] = 0
print(df['price'].mean())
mean_price = numeric_prices.mean()
print(mean_price)

print(df.price)