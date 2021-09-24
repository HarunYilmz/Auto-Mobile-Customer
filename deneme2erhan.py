#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#verileri yükleme
train_data = pd.read_csv('train-set.csv').dropna()
test_data = pd.read_csv('test-set.csv').dropna()

x = train_data.iloc[:,2:10].dropna()
y = train_data.iloc[:,-1:].dropna()
X = x.values
Y = y.values

#veri ön işleme
ms = x.iloc[:,3:4]
m = x.iloc[:,:1]
g = x.iloc[:,2:3]
ss = x.iloc[:,-3:-2]
ct = x.iloc[:,-1:]

#Sütunlardaki unique değerleri çekme
msColumn = ms.Profession.unique()
print(msColumn)
ctColumn = ct.Category.unique()

#Kategorik sütunlar için OneHotEncoder uygulama
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer 


cct = make_column_transformer(
    (OneHotEncoder(), ['Profession']), remainder="passthrough")
ms2 = cct.fit_transform(ms).todense()

ms2 = pd.DataFrame(ms2)
ms3 = ms2.rename(columns = { 0 : 'Artist', 
                             1 : 'Doctor', 
                             2 : 'Engineer', 
                             3 : 'Entertainment', 
                             4 : 'Executive',
                             5 : 'Healthcare',
                             6 : 'Homemaker', 
                             7 : 'Lawyer', 
                             8 : 'Marketing'      })
#index eşitleme işlemleri
awd = np.arange(0,6665,1) # ms3 indexini düzenlemek için oluşturulan array
ms4 = ms.set_index(np.arange(0,6665,1)) # ms3'ün indexleri düzenlenmiş hali
m2 = m.set_index(np.arange(0,6665,1))
g2 = g.set_index(np.arange(0,6665,1))
ss2 = ss.set_index(np.arange(0,6665,1))
ct2 = ct.set_index(np.arange(0,6665,1))

# married sütunu için label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
[m2,g3] = le.fit_transform([m,g2])
# graduated sütunu için label encoding






