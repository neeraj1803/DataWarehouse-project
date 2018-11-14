# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:24:56 2018

@author: neera
"""
# importing libraries 
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')    # Loading Data from CSV file
X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns


# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))  # using single linkage technique

# Run until finding best number of clusters then run below code
    
# applying hierarchical clustering to our dataset 
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward') # number of clusters=2
PredictH=Hierarchical.fit_predict(X)     


# visualization of our hierarchical clustering
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')

plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.ylabel('Age')
plt.legend()
plt.show()
