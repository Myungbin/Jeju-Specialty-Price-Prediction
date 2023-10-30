import pandas as pd

train = pd.read_csv(r'./data/train.csv')
test = pd.read_csv(r'./data/test.csv')
international_trade = pd.read_csv(r'./data/international_trade.csv')

print(international_trade.head())