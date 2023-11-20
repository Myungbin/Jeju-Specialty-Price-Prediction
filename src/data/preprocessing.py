import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.feature_extraction import group_season, holiday, cyclical_feature, determine_harvest_weight


class DataPreprocessing:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    @staticmethod
    def label_encode(train, test):
        categorical_col = train.select_dtypes(include=['object']).columns.tolist()
        print(f"Category columns: {categorical_col}")

        for i in categorical_col:
            le = LabelEncoder()
            train[i] = le.fit_transform(train[i])
            test[i] = le.transform(test[i])

        return train, test

    @staticmethod
    def remove_outliers(train):
        print('Remove outliers')
        train.loc[(train['Weekday'] == 6) & (train['price(원/kg)'] >= 0), 'price(원/kg)'] = 0
        
        # item_id에서 평균가격이 낮은 데이터를 0으로 치환
        train.loc[(train['item_id']=="RD_C_S") & (train['price(원/kg)'] > 0), ['price(원/kg)', 'supply(kg)']] = 0

        train.loc[(train['item_id']=="BC_C_S") & (train['price(원/kg)'] > 0), ['price(원/kg)', 'supply(kg)']] = 0
        train.loc[(train['item_id']=="BC_B_S") & (train['price(원/kg)'] > 0), ['price(원/kg)', 'supply(kg)']] = 0
        
        train.loc[(train['item_id']=="CR_E_S") & (train['price(원/kg)'] > 0), ['price(원/kg)', 'supply(kg)']] = 0
        train.loc[(train['item_id']=="CR_D_S") & (train['price(원/kg)'] > 0), ['price(원/kg)', 'supply(kg)']] = 0

        train.loc[(train['item_id']=="CB_A_S") & (train['price(원/kg)'] > 0), ['price(원/kg)', 'supply(kg)']] = 0
        
        # item_location에서 평균 가격이 낮은 데이터를 0으로 치환        
        train.loc[(train['item_location']=="CRS") & (train['price(원/kg)'] > 0), ['price(원/kg)', 'supply(kg)']] = 0
        
        # item_month_Weekday에서 평균 가격이 낮은 데이터 0으로 치환
        condition = train.groupby('item_month_Weekday')['price(원/kg)'].mean() < 50
        indexes_to_replace = condition[condition].index
        train.loc[train['item_month_Weekday'].isin(indexes_to_replace), ['price(원/kg)', 'supply(kg)']] = 0
        return train

    @staticmethod
    def preprocessing(data):
        print('Preprocessing Start')
        # time feature
        data['year'] = data['timestamp'].apply(lambda x: int(x[0:4]))
        data['month'] = data['timestamp'].apply(lambda x: int(x[5:7]))
        data['day'] = data['timestamp'].apply(lambda x: int(x[8:10]))
        
        data['Weekday'] = pd.to_datetime(data['timestamp']).dt.weekday
        data['is_weekend'] = data['Weekday'].apply(lambda x: 1 if x >= 6 else 0)
        data['year'] = data['year'] - 2019
        data['season'] = group_season(data)
        data['holiday'] = holiday(data)        
        # item feature
        data['item_id'] = data.ID.str[0:6]
        data['total_value_month'] = data['item_id'] + data['month'].astype(str)
        data['item_location'] = data['item']+data['location']
        data['item_corporation'] = data['item']+data['corporation']
        data['item_month_Weekday'] = data['item'].astype(str) + "_" + data['month'].astype(str) + data['Weekday'].astype(str)
        data['item_month_corp'] = data['item']+data['month'].astype(str)+data['corporation']
        data['location_cooperation'] = data['location']+data['corporation']
        data['location_cooperation_month'] = data['location']+data['corporation']+data['month'].astype(str)
        data['item_month_day'] = data['item'].astype(str) + "_" + data['month'].astype(str) + data['day'].astype(str)
        data['month_day'] = data['month'].astype(str) + "_" + data['day'].astype(str)

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['harvest_weight'] = data.apply(lambda row: determine_harvest_weight(row['item'], row['timestamp'].month), axis=1)

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

def post_preprocessing(test, submission):
    idx_list = test[(test['Weekday'] == 6)].index
    submission.loc[idx_list, 'answer'] = 0 # Weekday == 6 (일요일)이면 가격 0원
    submission['answer'] = submission['answer'].apply(lambda x: max(0, x)) # 가격에 음수가 있다면 가격 0원으로 변경
    return submission