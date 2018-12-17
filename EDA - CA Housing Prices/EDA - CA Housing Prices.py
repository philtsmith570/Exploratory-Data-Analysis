# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:17:50 2018

@author: philt
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('bmh')

df = pd.read_csv('train.csv')

#print(df.head())
#print(df.info())

# df.count() does not include NaN values
df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]
del df2['Id']
print("list of dropped columns:", end=" ")
for c in df.columns:
    if c not in df2.columns:
        print(c, end=", ")
    
print("\n")
df = df2

# Plot housing price distribution 
print('Sales Price Stats: ', df['SalePrice'].describe())
plt.figure(figsize=(18, 8))
sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4})


list(set(df.dtypes.tolist()))

df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

# Plot the histogram of all features
matplotlib.style.use('ggplot')
params = {'axes.titlesize':'8',
          'xtick.labelsize':'8',
          'ytick.labelsize':'8'}
matplotlib.rcParams.update(params)

df_num.hist(figsize=(18, 9), bins=50, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()

# Find corrlation between features and Sales Price
df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(
        ascending=False)
print("There are {} strongly correlated values with SalePrice:\n{}".format(
        len(golden_features_list), golden_features_list))


# Plot Corraltion with SalePrice
for i in range(0, len(df_num.columns), 10):
    sns.pairplot(data=df_num, 
                 x_vars=df_num.columns[i:i+10], y_vars=['SalePrice'])

plt.tight_layout()
plt.show()
#
#sns.set(style='ticks', color_codes=True)
#g = sns.FacetGrid(df_num, col='GarageArea', col_wrap=4, size=2, ylim=(0, 10))
#g.map(sns.pairplot, df_num, "SalePrice", color=".3", ci=None);
#g = sns.FacetGrid(df_num, col='GarageArea', row='SalePrice')
#sns.pairplot(data=df_num, x_vars=df_num.columns[i:i+5], y_vars=['SalePrice'])


