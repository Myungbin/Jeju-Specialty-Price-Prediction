import warnings

import holidays
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
train = pd.read_csv(r'./data/train.csv')
international_trade = pd.read_csv(r'./data/international_trade.csv')
test = pd.read_csv(r'./data/test.csv')


def group_season(df):
    df.loc[(df['month'] == 3) | (df['month'] == 4)
           | (df['month'] == 5), 'season'] = '봄'
    df.loc[(df['month'] == 6) | (df['month'] == 7)
           | (df['month'] == 8), 'season'] = '여름'
    df.loc[(df['month'] == 9) | (df['month'] == 10)
           | (df['month'] == 11), 'season'] = '가을'
    df.loc[(df['month'] == 12) | (df['month'] == 1)
           | (df['month'] == 2), 'season'] = '겨울'
    return df['season']


def holiday(df):
    kr_holidays = holidays.KR()
    df['holiday'] = df.timestamp.apply(
        lambda x: 'holiday' if x in kr_holidays else 'non-holiday')
    return df['holiday']


train['year'] = train['timestamp'].apply(lambda x: int(x[0:4]))
train['month'] = train['timestamp'].apply(lambda x: int(x[5:7]))
train['day'] = train['timestamp'].apply(lambda x: int(x[8:10]))
train['Weekday'] = pd.to_datetime(train['timestamp']).dt.weekday
train['is_weekend'] = train['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

test['year'] = test['timestamp'].apply(lambda x: int(x[0:4]))
test['month'] = test['timestamp'].apply(lambda x: int(x[5:7]))
test['day'] = test['timestamp'].apply(lambda x: int(x[8:10]))
test['Weekday'] = pd.to_datetime(test['timestamp']).dt.weekday
test['is_weekend'] = test['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

train['season'] = group_season(train)
test['season'] = group_season(test)

train['holiday'] = holiday(train)
test['holiday'] = holiday(test)

x = train.drop(columns=['ID', 'timestamp', 'supply(kg)', 'price(원/kg)'])
y = train['price(원/kg)']

x_test = test.drop(columns=['ID', 'timestamp'])

qual_col = ['item', 'corporation', 'location',
            'season', 'holiday', 'total_item_value']

for i in qual_col:
    le = LabelEncoder()
    x[i] = le.fit_transform(x[i])
    x_test[i] = le.transform(x_test[i])

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=1103)
xgb = XGBRegressor()
xgb.fit(x, y)
xgb_pred = xgb.predict(x_test)
submission = pd.read_csv(r'data\sample_submission.csv')
submission['answer'] = xgb_pred
submission.to_csv('./xgb_submission.csv', index=False)
