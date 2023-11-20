import holidays
import numpy as np


def group_season(df):
    df.loc[(df['month'] == 3) | (df['month'] == 4) | (df['month'] == 5), 'season'] = '봄'
    df.loc[(df['month'] == 6) | (df['month'] == 7) | (df['month'] == 8), 'season'] = '여름'
    df.loc[(df['month'] == 9) | (df['month'] == 10) | (df['month'] == 11), 'season'] = '가을'
    df.loc[(df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2), 'season'] = '겨울'
    return df['season']


def holiday(df):
    kr_holidays = holidays.KR()
    df['holiday'] = df.timestamp.apply(lambda x: 'holiday' if x in kr_holidays else 'non-holiday')
    return df['holiday']


def cyclical_feature(df, time=12):
    df['sin_time'] = np.sin(2 * np.pi * df.month / time)
    df['cos_time'] = np.cos(2 * np.pi * df.month / time)

def determine_harvest_weight(item, month):
    harvest_times = {
    'TG': {'main': [(10, 1)]},  # 감귤: 10월부터 이듬해 1월까지
    'BC': {'main': [(4, 6), (9, 11)]},  # 브로콜리: 4월-6월, 9월-11월
    'RD': {'main': [(5, 6), (11, 12)]},  # 무: 5월, 11월
    'CR': {'main': [(7, 8), (10, 11)]},  # 당근: 7월-8월, 10월-12월
    'CB': {'main': [(6, 6), (11, 11)]}  # 양배추: 6월, 11월
}
    main_harvest = harvest_times[item]['main']
    for start, end in main_harvest:
        if start <= month <= end:
            return 1
    return 0