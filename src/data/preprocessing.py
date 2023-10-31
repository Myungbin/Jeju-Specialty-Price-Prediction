import pandas as pd
from src.data.feature_extraction import group_season, holiday, cyclical_feature
from sklearn.preprocessing import LabelEncoder


def preprocessing(train, test):
    train['year'] = train['timestamp'].apply(lambda x: int(x[0:4]))
    train['month'] = train['timestamp'].apply(lambda x: int(x[5:7]))
    train['day'] = train['timestamp'].apply(lambda x: int(x[8:10]))
    train['Weekday'] = pd.to_datetime(train['timestamp']).dt.weekday
    train['is_weekend'] = train['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    train['year'] = train['year'] - 2019
    train['total_item_value'] = train['item'] + train['corporation'] + train['location']
    train['location_month'] = train['location'].astype(str) + "_" + train['month'].astype(str)
    # cyclical_feature(train)

    test['year'] = test['timestamp'].apply(lambda x: int(x[0:4]))
    test['month'] = test['timestamp'].apply(lambda x: int(x[5:7]))
    test['day'] = test['timestamp'].apply(lambda x: int(x[8:10]))
    test['Weekday'] = pd.to_datetime(test['timestamp']).dt.weekday
    test['is_weekend'] = test['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    test['year'] = test['year'] - 2019
    test['total_item_value'] = test['item'] + test['corporation'] + test['location']
    test['location_month'] = test['location'].astype(str) + "_" + test['month'].astype(str)
    # cyclical_feature(test)

    train['season'] = group_season(train)
    test['season'] = group_season(test)

    train['holiday'] = holiday(train)
    test['holiday'] = holiday(test)

    Q1 = train['price(원/kg)'].quantile(0.25)
    Q3 = train['price(원/kg)'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train = train[(train['price(원/kg)'] >= lower_bound) & (train['price(원/kg)'] <= upper_bound)]

    x = train.drop(columns=['ID', 'timestamp', 'supply(kg)', 'price(원/kg)'])
    y = train['price(원/kg)']

    x_test = test.drop(columns=['ID', 'timestamp'])

    qual_col = ['item', 'corporation', 'location', 'season', 'holiday',
                'Weekday', 'is_weekend', 'total_item_value', 'location_month']

    for i in qual_col:
        le = LabelEncoder()
        x[i] = le.fit_transform(x[i])
        x_test[i] = le.transform(x_test[i])

    return x, y, x_test
