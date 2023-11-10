import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.feature_extraction import group_season, holiday, cyclical_feature


class DataPreprocessing:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    @staticmethod
    def label_encode(train, test):
        categorical_col = ['item', 'corporation', 'location', 'season', 'holiday', 'total_item_value',
                           'item_month_Weekday', 'corporation_weekday', 'item_season', 'year_season']

        for i in categorical_col:
            le = LabelEncoder()
            train[i] = le.fit_transform(train[i])
            test[i] = le.transform(test[i])

        return train, test

    @staticmethod
    def remove_outliers(train):
        print('Remove outliers')
        Q1 = train['price(원/kg)'].quantile(0.25)
        Q3 = train['price(원/kg)'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 3 * IQR
        train = train[(train['price(원/kg)'] >= lower_bound) & (train['price(원/kg)'] <= upper_bound)]
        return train

    @staticmethod
    def preprocessing(data):
        print('Preprocessing Start')
        # time feature
        data['year'] = data['timestamp'].apply(lambda x: int(x[0:4]))
        data['month'] = data['timestamp'].apply(lambda x: int(x[5:7]))
        data['Weekday'] = pd.to_datetime(data['timestamp']).dt.weekday
        data['is_weekend'] = data['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
        data['year'] = data['year'] - 2019
        data['season'] = group_season(data)
        data['holiday'] = holiday(data)
        # cyclical_feature(data)

        # item feature
        data['total_item_value'] = data['item'] + data['corporation'] + data['location']
        data['item_month_Weekday'] = data['item'].astype(str) + "_" + data['month'].astype(str) + data[
            'Weekday'].astype(str)
        data['corporation_weekday'] = data['corporation'].astype(str) + "_" + data['Weekday'].astype(str)
        data['item_season'] = data['item'].astype(str) + "_" + data['season'].astype(str)
        data['year_season'] = data['year'].astype(str) + "_" + data['season'].astype(str)

        return data

    def fit(self):
        self.train = self.preprocessing(self.train)
        self.test = self.preprocessing(self.test)

        self.train = self.remove_outliers(self.train)

        x_train = self.train.drop(columns=['ID', 'timestamp', 'supply(kg)', 'price(원/kg)'])
        y_train = self.train['price(원/kg)']
        x_test = self.test.drop(columns=['ID', 'timestamp'])

        x_train, x_test = self.label_encode(x_train, x_test)

        return x_train, y_train, x_test

#
# def preprocessing(train, test):
#     train['year'] = train['timestamp'].apply(lambda x: int(x[0:4]))
#     train['month'] = train['timestamp'].apply(lambda x: int(x[5:7]))
#     train['day'] = train['timestamp'].apply(lambda x: int(x[8:10]))
#     train['Weekday'] = pd.to_datetime(train['timestamp']).dt.weekday
#     train['is_weekend'] = train['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
#     train['year'] = train['year'] - 2019
#     train['total_item_value'] = train['item'] + train['corporation'] + train['location']
#     train['location_month'] = train['location'].astype(str) + "_" + train['month'].astype(str)
#     # cyclical_feature(train)
#
#     test['year'] = test['timestamp'].apply(lambda x: int(x[0:4]))
#     test['month'] = test['timestamp'].apply(lambda x: int(x[5:7]))
#     test['day'] = test['timestamp'].apply(lambda x: int(x[8:10]))
#     test['Weekday'] = pd.to_datetime(test['timestamp']).dt.weekday
#     test['is_weekend'] = test['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
#     test['year'] = test['year'] - 2019
#     test['total_item_value'] = test['item'] + test['corporation'] + test['location']
#     test['location_month'] = test['location'].astype(str) + "_" + test['month'].astype(str)
#     # cyclical_feature(test)
#
#     train['season'] = group_season(train)
#     test['season'] = group_season(test)
#
#     train['holiday'] = holiday(train)
#     test['holiday'] = holiday(test)
#
#     Q1 = train['price(원/kg)'].quantile(0.25)
#     Q3 = train['price(원/kg)'].quantile(0.75)
#     IQR = Q3 - Q1
#
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     train = train[(train['price(원/kg)'] >= lower_bound) & (train['price(원/kg)'] <= upper_bound)]
#
#     x = train.drop(columns=['ID', 'timestamp', 'supply(kg)', 'price(원/kg)'])
#     y = train['price(원/kg)']
#
#     x_test = test.drop(columns=['ID', 'timestamp'])
#
#     qual_col = ['item', 'corporation', 'location', 'season', 'holiday',
#                 'Weekday', 'is_weekend', 'total_item_value', 'location_month']
#
#     for i in qual_col:
#         le = LabelEncoder()
#         x[i] = le.fit_transform(x[i])
#         x_test[i] = le.transform(x_test[i])
#
#     return x, y, x_test
