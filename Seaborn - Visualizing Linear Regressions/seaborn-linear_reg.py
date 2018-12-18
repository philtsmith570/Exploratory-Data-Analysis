# -*- coding: utf-8 -*-
# seaborn-linear_reg.py
"""
Created on Sat Mar 31 21:24:21 2018

@author: philt
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

np.random.seed(sum(map(ord, "regression")))

tips = sns.load_dataset("tips")

# Regression Plot
sns.regplot(x="total_bill", y="tip", data=tips)

# lmplot Total Bill vs. Tip
sns.lmplot(x="total_bill", y="tip", data=tips)

# lmplot Size Vs Tip
sns.lmplot(x="size", y="tip", data=tips)
sns.lmplot(x="size", y="tip", data=tips, x_jitter=0.05)

# lmplot - collapse over each observation with central tendency and confidence interval
sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean)

# Load anscombe dataset
anscombe = sns.load_dataset("anscombe")


''' Plot against non-linear looking data '''

# Note: ci= None turns off the confidence interval.  scatter "s": 80
# Creates a larger point on graph

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
           ci=None, scatter_kws={"s": 80})

##sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"), ci=None,
#           scatter_kws={"s": 80})

## Use order to fit the data
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"), order=2,
           ci=None, scatter_kws={"s": 80})


''' Outlier issue.  Using Robust for linear regression '''
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           ci=None, scatter_kws={"s": 80})

#sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
#           robust=True, ci=None, scatter_kws={"s": 80})


''' Looking at binary variable.  Regression is possible, but not plausible '''
tips["big_tip"] = (tips.tip / tips.total_bill) > 0.15

sns.lmplot(x="total_bill", y="big_tip", data=tips, y_jitter=0.03)


# Use logistic=True for a fit of the binary data
sns.lmplot(x="total_bill", y="big_tip", data=tips, y_jitter=0.03,
           logistic=True)

# Note:  The logistic regression as well as the robust regression are both 
# computationally expense.  Turning off ci will help reduce the compute time.


# using lowess - LOWESS (locally weighted scatterplot smoothing)
sns.lmplot(x="total_bill", y="tip", data=tips,
           lowess=True)


# The residplot is useful for checking whether the simple regression model is
# appropriate for a dataset.  It fits, the nremoves a simple linear regression, 
# and then plots the residuals

# This data shows no structure, i.e. appears random.  Good for linear regression.
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
             scatter_kws={"s": 80})

# Structure means not a good candidate for linear regression
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
             scatter_kws={"s": 80})

''' Conditioning on other Variables '''

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)

# Use markers to create different markers for each conditional variable
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1")

# To add another variable, you can draw multiple “facets” which each level 
# of the variable appearing in the rows or columns of the grid:
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);

sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips);

''' Controlling size and shape of the plot '''

'''Before we noted that the default plots made by regplot() and lmplot() look 
the same but on axes that have a different size and shape. This is because 
regplot() is an “axes-level” function draws onto a specific axes. This means 
that you can make multi-panel figures yourself and control exactly where the 
regression plot goes. If no axes object is explictly provided, it simply uses 
the “currently active” axes, which is why the default plot has the same size 
and shape as most other matplotlib functions. To control the size, you need to 
create a figure object yourself. '''

f, ax =plt.subplots(figsize=(5,6))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax)

# In contrast, the size and shape of the lmplot() figure is controlled through 
# the FacetGrid interface using the size and aspect parameters, which apply to 
# each facet in the plot, not to the overall figure itself:

sns.lmplot(x="total_bill", y="tip", col="day", data=tips, col_wrap=2, height=3)

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           aspect=0.5)

''' Plotting a regression in other contexts '''

''' A few other seaborn functions use regplot() in the context of a larger, 
more complex plot. The first is the jointplot() function that we introduced in 
the distributions tutorial. In addition to the plot styles previously discussed, 
jointplot() can use regplot() to show the linear regression fit on the joint 
axes by passing kind="reg": '''

sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg")

''' Using the pairplot() function with kind="reg" combines regplot() and 
PairGrid to show the linear relationship between variables in a dataset. Take 
care to note how this is different from lmplot(). In the figure below, the two 
axes don’t show the same relationship conditioned on two levels of a third 
variable; rather, PairGrid() is used to show multiple relationships between 
different pairings of the variables in a dataset.  '''

sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             height=5, aspect=.8, kind="reg")

''' Like lmplot(), but unlike jointplot(), conditioning on an additional 
categorical variable is built into pairplot() using the hue parameter. '''
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", height=5, aspect=0.8, kind="reg")
