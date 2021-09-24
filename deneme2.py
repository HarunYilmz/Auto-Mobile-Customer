#Kütüphaneler
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt


#verileri yükleme
train_data = pd.read_csv('train-set.csv')
test_data = pd.read_csv('test-set.csv')

x = train_data.iloc[:,2:10]
y = train_data.iloc[:,-1:]
X = x.values
Y = y.values

#veri ön işleme

ms = x.iloc[:,3:4]
m = x.iloc[:,:1]
g = x.iloc[:,2:3]
ss = x.iloc[:,-3:-2]
ct = x.iloc[:,-1:]

#dönüştürme
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ms = ohe.fit_transform(ms).toarray()
print(ms)


sonucmeslek = pd.DataFrame(data=ms ,index = range(8068), columns= ['Healthcare','Engineer','Lawyer','Entertainment','Artist','Executive','Doctor','Homemaker','Marketing','nan'])
print(sonucmeslek)
