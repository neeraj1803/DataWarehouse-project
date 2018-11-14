T#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

# importing our csv dataset

mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[6,13]].values  # Loading Age and Gender

# Find out the best number of clusters

Array=[]     # to store sum of squares within the groups

for i in range(1,14):  # considering a random of 15 clusters
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0) 
    kmeans.fit(X)
    Array.append(kmeans.inertia_)    # inertia --> Sum of squared distances of samples to their closest cluster center

plt.plot(range(1,14),Array)    
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squares within groups')
plt.show()   # Displays the number of clusters we need to consider or optimal clusters we need to consider

# K-Means clustering algorithm on Rider age and Gender using Uber cabs data

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)  

# fit function will give output of kmeans but fit_predict will give the cluster index for each sample
Y=kmeans.fit_predict(X)

plt.scatter(X[Y == 0,0], X[Y == 0,1],s=25,c='red',label='cluster 1')   #s --> zoom level
plt.scatter(X[Y == 1,0], X[Y == 1,1],s=25,c='blue',label='cluster 2')
plt.scatter(X[Y == 2,0], X[Y == 2,1],s=25,c='magenta',label='cluster 3')
plt.scatter(X[Y == 3,0], X[Y == 3,1],s=25,c='cyan',label='cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=25,c='yellow',label='Centroid')
plt.title('K-Means Clustering')
plt.xlabel('Gender')
plt.ylabel('Rider Age')
plt.legend()
plt.show() 