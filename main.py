import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# read file, print first 5 rows of the dataset
df = pd.read_csv('Rainfall.csv')
print(df.head())

# print size of dataset (rows, columns)
print(df.shape)

# print columns and their data types, info of null values
print(df.info())

# print statistical summary of the dataset
print(df.describe().T)

# print number of null values in each column (1 for winddirection and 1 for windspeed)
print(df.isnull().sum())

# print column names
print(df.columns)

# remove unnecessary spaces in the names of the columns
df.rename(str.strip,
          axis='columns', 
          inplace=True)
print(df.columns)

# Checking if the column contains any null values, if yes then fill it with mean of the column
for col in df.columns:
  if df[col].isnull().sum() > 0:
    val = df[col].mean()
    df[col] = df[col].fillna(val)

# print number of null values in each column   
print(df.isnull().sum().sum())

# shows the distribution of values ​​in the rainfall column with a pie chart 
plt.pie(df['rainfall'].value_counts().values,
        labels = df['rainfall'].value_counts().index,
        autopct='%1.1f%%')
plt.show()





# i will check !!! 
print(df.groupby('rainfall').mean())

features = list(df.select_dtypes(include = np.number).columns)
features.remove('day')
print(features)

plt.subplots(figsize=(15,8))

for i, col in enumerate(features):
  plt.subplot(3,4, i + 1)
  sb.distplot(df[col])
plt.tight_layout()
plt.show()

