# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:37:55 2018

@author: neera
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *
import matplotlib.pyplot as plt

df=pd.read_csv("classification_problem_Full.csv")

print(df.head(10))
print("CATE",df["CATEGORY*"].value_counts())

#CATEGORY
#ct_dis=df["CATEGORY*"].value_counts()
#figure(2)
#rects=plt.bar(range(1,len(ct_dis.index)+1),ct_dis.values)
#plt.title("Category DISTRIBUTE")
#plt.xlabel("Category")
#plt.ylabel("Quantity")
#plt.xticks(range(1,len(ct_dis.index)+1),ct_dis.index)
#plt.grid()
#autolabel(rects)
#plt.savefig("./ct_dis_fig")

g = sns.factorplot(x="PURPOSE*", y="Gender", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")