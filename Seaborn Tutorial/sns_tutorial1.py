# -*- coding: utf-8 -*-
# 
# sns_tutorial1.py
"""
Created on Sun Feb 18 17:09:14 2018

@author: philt
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Load dataset
df = pd.read_csv("pokemon.csv", encoding = "ISO-8859-1", index_col=0)

###Recommneded way
#plt.figure()
#sns.lmplot(x='Attack', y='Defense', data=df,
#           fit_reg=False, # No regression line
#           hue='Stage') # Color by evolution stage
#
#plt.xlim(0, None)
#plt.ylim(0, None)

###Boxplot
#
## Preformat data in pandas - Remove total, stage, legendary
#plt.figure()
#stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)
#sns.boxplot(data=df)
#
## Set theme
#sns.set_style('whitegrid')
#
## Color Palette
pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

## Violin plot
plt.figure()
sns.violinplot(x='Type_1', y='Attack', data=df, palette=pkmn_type_colors)


## Swarm plot with Pokemon w/ color palette
plt.figure()
sns.swarmplot(x='Type_1', y='Attack', data=df, palette=pkmn_type_colors)
# Set figure size with matplotlib
plt.figure(figsize=(12, 8))


# Create plot

sns.violinplot(x='Type_1',
               y='Attack', 
               data=df, 
               inner=None, # Remove the bars inside the violins
               palette=pkmn_type_colors)


sns.swarmplot(x='Type_1', 
              y='Attack', 
              data=df, 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent

# Set title with matplotlib
plt.title('Attack by Type')


# Swarmplot with melted_df
stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)
melted_df = pd.melt(stats_df, id_vars=["Name", "Type_1", "Type_2", ],  #var to keep
                    var_name="Stat")  # Name of melted variable

sns.swarmplot(x='Stat', 
              y='value', 
              data=melted_df, 
              hue='Type_1',
              split=True, # Seperate the points by hue
              palette=pkmn_type_colors)

#Adjust the y-axis
plt.ylim(0,260)


## Heat map - Heat maps help you visualize matrix-like data
#Calculate the correlations
corr = stats_df.corr()
sns.heatmap(corr, cmap='Reds')
plt.legend(bbox_to_anchor=(1,1), loc=2)


## Distribution Plot (Histogram)
sns.distplot(df.Attack)


## Bar Plot 
sns.countplot(x='Type_1', data=df, palette=pkmn_type_colors)
plt.xticks(rotation=-45)


## Factor Plot - makes it easy to separate plots by category
g = sns.factorplot(x='Type_1',
                   y='Attack',
                   data=df,
                   hue='Stage', # Color by stage
                   col='Stage', # Separate by Stage
                   kind='swarm') # Swarm plot

g.set_xticklabels(rotation=-45)  # Note: you can only do this once.  i.e., last plot


## Density Plot - Contour Plot
# Displays Distribution between two variables
fig, ax = plt.subplots()
sns.kdeplot(df.Attack, df.Defense, ax=ax)
# Added scatterplot
sns.regplot(x='Attack', y='Defense', data=df,
           fit_reg=False, # No regression line
           scatter_kws={'s':2},  # Change size of data points
           ax=ax) # Color by evolution stage

plt.xlim(0, None)
plt.ylim(0, None)

### Joint Distribution Plot
##  Combine infornation from scatter plots and histograms to give detailed 
##  information for Bi-variate distributions
#sns.jointplot(x='Attack', y='Defense', data=df)
