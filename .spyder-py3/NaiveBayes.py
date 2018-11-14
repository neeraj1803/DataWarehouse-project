# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:31:27 2018

@author: neera
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values  #Loading Gender,Category,Purpose features
y=mydata.iloc[:,[3]].values      # Loading Age feature

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)  # Train:test=75:25


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))
# visualize the training data

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j],X_set[y_set==j],
    #            c=ListedColormap(('red','green'))(i),label=j)

#plt.scatter(X_set[:,0],y_train[:,1])
#plt.bar(np.arange(0,3),height=20)
#plt.xticks(np.arange(0,3),height=15)
#plt.title('Naive Bayes Visualization')
#plt.show()    

PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="PURPOSE*", data=mydata,
                   size=10, kind="bar", palette="muted")

GendervsAge = sns.factorplot(x="Gender", y="Age", hue="Gender", data=mydata,
                   size=10, kind="bar", palette="muted")
