
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

data_path=' '
data=pd.read_csv(data_path)
data.columns=[i.lower() for i in data.columns]
data.drop(data.loc[data['season'].str.contains('1993|1994',regex=True)].index,inplace=True)
data['date']=pd.to_datetime(data['date'])
data['year']=data.date.apply(lambda x : x.year)
data['month']=data.date.apply(lambda x : x.month)
data['day']=data.date.apply(lambda x : x.day)
data.reset_index(drop=True,inplace=True)
oe=OrdinalEncoder(categories=[['H','D','A'],['H','D','A']])
data[['htr','ftr']]=oe.fit_transform(data[['htr','ftr']])
data.drop(['div','season'],axis=1,inplace=True)
data.sort_values('date',inplace=True)
# team_period_between_matches
data['tpbm']=data.groupby('hometeam')['date'].apply(lambda x:x.diff()).apply(lambda x :x.days) # or .apply(lambda x: x-x.shift(1))
data['tpbm'].fillna(0,inplace=True)