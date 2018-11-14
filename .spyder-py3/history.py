# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Wed Apr 18 11:26:28 2018)---
runfile('C:/Users/neera/.spyder-py3/temp.py', wdir='C:/Users/neera/.spyder-py3')
mydata
runfile('C:/Users/neera/.spyder-py3/temp.py', wdir='C:/Users/neera/.spyder-py3')
mydata.head
mydata[ ,1]
mydata[1,1]
runfile('C:/Users/neera/.spyder-py3/temp.py', wdir='C:/Users/neera/.spyder-py3')
mydata.head()
plt.plot(mydata[:2])
mydata.head()
mydata[:2]
mydata[2]
mydata[0:2]
mydata[category]
mydata[:, :2]
print(mydata['CATEGORY'])
mydata['CATEGORY']
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[5,13]]
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[5,13]].values
wcss=[]

for i in range[1,16]:
    kmeans=KMeans(m_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append[kmeans.inertia_]

wcss=[]

for i in range[1,13]:
    kmeans=KMeans(m_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append[kmeans.inertia_]

wcss=[]

for i in range[1,14]:
    kmeans=KMeans(m_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append[kmeans.inertia_]

from sklearn.cluster import KMeans
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[5,13]].values

# Find out the best number of clusters

wcss=[]

for i in range[1,14]:
    kmeans=KMeans(m_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append[kmeans.inertia_]

wcss=[]

for i in range(1,14):
    kmeans=KMeans(m_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append[kmeans.inertia_]

wcss=[]

for i in range(1,14):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append[kmeans.inertia_]

mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[6,13]].values
wcss=[]

for i in range(1,14):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append[kmeans.inertia_]

wcss=[]

for i in range(1,14):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,16),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
plt.plot(range(1,14),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
runfile('C:/Users/neera/.spyder-py3/temp.py', wdir='C:/Users/neera/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


plt.title('Naive Bayes Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show() 
runfile('C:/Users/neera/.spyder-py3/NaiveBayes.py', wdir='C:/Users/neera/.spyder-py3')
runfile('C:/Users/neera/.spyder-py3/temp.py', wdir='C:/Users/neera/.spyder-py3')
runfile('C:/Users/neera/.spyder-py3/NaiveBayes.py', wdir='C:/Users/neera/.spyder-py3')

## ---(Thu Apr 19 09:00:19 2018)---
runfile('C:/Users/neera/.spyder-py3/NaiveBayes.py', wdir='C:/Users/neera/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
runfile('C:/Users/neera/.spyder-py3/NaiveBayes.py', wdir='C:/Users/neera/.spyder-py3')
"""
Created on Thu Apr 19 09:08:39 2018

@author: neera
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:31:27 2018

@author: neera
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

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

y_pred=classifier.predict(X_test)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# visualize the training data

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


plt.title('Naive Bayes Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
runfile('C:/Users/neera/.spyder-py3/NaiveBayes.py', wdir='C:/Users/neera/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
runfile('C:/Users/neera/.spyder-py3/NaiveBayes.py', wdir='C:/Users/neera/.spyder-py3')
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

## ---(Thu Apr 19 10:32:34 2018)---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('DM_PR8.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('DM_PR8.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('DM_PR8.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
mydata[:,[2]]
mydata[:,[1,2]]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
x=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


plt.title('Naive Bayes Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


plt.title('Naive Bayes Visualization')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_test[:,0],X_test[:,1],c=Z,cmap=cmap_bold)
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

matplotlib.scatter(X_train,X_test)
matplotlib.title('Naive Bayes Visualization')
matplotlib.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_test[:,0],X_test[:,1],cmap=cmap_bold)
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

matplotlib.scatter(X_train,X_test)
matplotlib.title('Naive Bayes Visualization')
matplotlib.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_test[:,0],X_test[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

matplotlib.scatter(X_train,X_test)
matplotlib.title('Naive Bayes Visualization')
matplotlib.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_test[:,0],X_test[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_set[:,0],y_set[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_set[:,0],y_set[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
plt.scatter(X_set[:,0],y_set[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()


from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[:,0],y_set[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train,X_test)
plt.title('Naive Bayes Visualization')
plt.show()

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train,y_train)
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train[:,0],y_pred[:,1])
plt.title('Naive Bayes Visualization')
plt.show()
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train[:,0],y_test[:,1])
plt.title('Naive Bayes Visualization')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict([1,1,1])
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
from scipy.cluster.hierarchy import linkage, dendogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram1(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78]])

plt.scatter(x[:,0],x[:,1],s=50)
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15],[30],[65],[24],[25],[28],[40],[45],[49],[51],[75],[24],[26],[28],[45],[56],[62],[47],[78]])
plt.scatter(x[:,0],x[:,1],s=50)
plt.scatter(x[:,0],s=50)
plt.scatter(x[:,0],x[:,1],s=50)
linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])
plt.scatter(x[:,0],x[:,1],s=50)
linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram=dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix.truncate_node='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix.truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix.truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix.truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram1= dendrogram(linkage_matrix.truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix.truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix.truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,10],[10.5,12],[15,20],[10,8]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15],[10.5],[15],[10]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15],[10.5],[15],[10]])

plt.scatter(x[:,0],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15],[10.5],[15],[10]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,30],[60,25],[15,78],[10,40]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,30],[60,25],[15,78],[10,40]])

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
classifier.predict([[1,1,1]])

## ---(Sat Apr 21 22:35:05 2018)---
mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area


classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area


classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area


classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area


classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)

## ---(Sun Apr 22 11:31:22 2018)---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train[:,[0,1,2],y_test[:,1])
plt.title('Naive Bayes Visualization')
plt.show()
plt.scatter(X_train[:,[0,1,2]],y_test[:,1])
plt.title('Naive Bayes Visualization')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area
from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,0,0]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,0,0]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training1=classifier.score(X_train,y_train)
print(accuracy_training1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training1=classifier.score(X_train,y_train)
accuracy_testing1=classifier.score(X_test,y_test)
print(accuracy_testing1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training1=classifier.score(X_train,y_train)
accuracy_testing1=classifier.score(X_test,y_test)
print(accuracy_training1)
print(accuracy_testing1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import svm
classifier=svm.SVC() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
"""
Created on Thu Apr 19 00:31:27 2018

@author: neera
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training=classifier.score(X_train,y_train)
accuracy_testing=classifier.score(X_test,y_test)
print(accuracy_training)
print(accuracy_testing)


# get output of test results

#y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[0,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training=classifier.score(X_train,y_train)
accuracy_testing=classifier.score(X_test,y_test)
print(accuracy_training)
print(accuracy_testing)


# get output of test results

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[0,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training=classifier.score(X_train,y_train)
accuracy_testing=classifier.score(X_test,y_test)
print(accuracy_training)
print(accuracy_testing)


# get output of test results

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[0,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training=classifier.score(X_train,y_train)
accuracy_testing=classifier.score(X_test,y_test)
print(accuracy_training)
print(accuracy_testing)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
accuracy_training=classifier.score(X_train,y_train)
accuracy_testing=classifier.score(X_test,y_test)
print(accuracy_training)
print(accuracy_testing)


# get output of test results

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[0,1,1]])
print(y_pred)
print("Accuracy on training data set",accuracy_training)
print("Accuracy on testing data set",accuracy_testing)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

plt.plot(X_train,y_train)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

plt.plot(X_train,y_train)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

plt.plot(X_train,y_train)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

plt.plot(X,y)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

plt.scatter(X,y)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

plt.scatter(X[:,0],y[:,1])
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[3,12,13,14]].values


plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[3,12,13,14]].values

plt.scatter(X[Y == 0,0], X[Y == 0,1],s=25,c='red',label='cluster 1')   #s --> zoom level
plt.scatter(X[Y == 1,0], X[Y == 1,1],s=25,c='blue',label='cluster 2')
plt.scatter(X[Y == 2,0], X[Y == 2,1],s=25,c='magenta',label='cluster 3')
plt.scatter(X[Y == 3,0], X[Y == 3,1],s=25,c='cyan',label='cluster 4')

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(x,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[3,12,13,14]].values

plt.scatter(X[Y == 0,0], X[Y == 0,1],s=25,c='red',label='cluster 1')   #s --> zoom level
plt.scatter(X[Y == 1,0], X[Y == 1,1],s=25,c='blue',label='cluster 2')
plt.scatter(X[Y == 2,0], X[Y == 2,1],s=25,c='magenta',label='cluster 3')
plt.scatter(X[Y == 3,0], X[Y == 3,1],s=25,c='cyan',label='cluster 4')

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(X,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[3,12,13,14]].values

plt.scatter(X[Y == 0,0], X[Y == 0,1],s=25,c='red',label='cluster 1')   #s --> zoom level
plt.scatter(X[Y == 1,0], X[Y == 1,1],s=25,c='blue',label='cluster 2')
plt.scatter(X[Y == 2,0], X[Y == 2,1],s=25,c='magenta',label='cluster 3')
plt.scatter(X[Y == 3,0], X[Y == 3,1],s=25,c='cyan',label='cluster 4')

plt.scatter(x[:,0],x[:,1],s=50)

linkage_matrix=linkage(X,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[3,12,13,14]].values

plt.scatter(X[Y == 0,0], X[Y == 0,1],s=25,c='red',label='cluster 1')   #s --> zoom level
plt.scatter(X[Y == 1,0], X[Y == 1,1],s=25,c='blue',label='cluster 2')
plt.scatter(X[Y == 2,0], X[Y == 2,1],s=25,c='magenta',label='cluster 3')
plt.scatter(X[Y == 3,0], X[Y == 3,1],s=25,c='cyan',label='cluster 4')

plt.scatter(X[:,0],X[:,1],s=50)

linkage_matrix=linkage(X,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[3,12,13,14]].values



plt.scatter(X[:,0],X[:,1],s=50)

linkage_matrix=linkage(X,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

x=np.array([[15,30],[60,25],[15,78],[10,40]])
#mydata=pd.read_csv('DM_PR8.csv')
#X=mydata.iloc[:,[3,12,13,14]].values



plt.scatter(X[:,0],X[:,1],s=50)

linkage_matrix=linkage(X,"single")

dendogram= dendrogram(linkage_matrix,truncate_mode='none')

plt.title('Hierarchical clustering')

plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[3,12,13,14]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.xlabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.xlabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,float[2,10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.xlabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[float(2),10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.xlabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[double(2),10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.xlabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[float(2),10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.xlabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[float(2),10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.xlabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[float(2),10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.ylabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.ylabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.ylabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)

# visualization of our hierarchical clustering

plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=50,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=50,c='green',label='cluster 3')


plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.ylabel('Age')
plt.legend()
plt.show()
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
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
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# applying hierarchical clustering to our dataset
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
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
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
#accuracy_training=classifier.score(X_train,y_train)
#accuracy_testing=classifier.score(X_test,y_test)
#print(accuracy_training)
#print(accuracy_testing)
y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")
#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area


classifier=tree.DecisionTreeClassifier(criterion='gini')
classifier.fit(X_train,y_train)
classifier.predict([[1,1,1]])
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *
import matplotlib.pyplot as plt

df=pd.read_csv("classification_problem_Full.csv")

print(df.head(10))
print("CATE",df["CATEGORY*"].value_counts())
ct_dis=df["CATEGORY*"].value_counts()
figure(2)
rects=plt.bar(range(1,len(ct_dis.index)+1),ct_dis.values)
plt.title("Category DISTRIBUTE")
plt.xlabel("Category")
plt.ylabel("Quantity")
plt.xticks(range(1,len(ct_dis.index)+1),ct_dis.index)
plt.grid()
autolabel(rects)
plt.savefig("./ct_dis_fig")
runfile('C:/Users/neera/.spyder-py3/testme.py', wdir='C:/Users/neera/.spyder-py3')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *
import matplotlib.pyplot as plt

df=pd.read_csv("classification_problem_Full.csv")

print(df.head(10))
print("CATE",df["CATEGORY*"].value_counts())

#CATEGORY
ct_dis=df["CATEGORY*"].value_counts()
figure(2)
rects=plt.bar(range(1,len(ct_dis.index)+1),ct_dis.values)
plt.title("Category DISTRIBUTE")
plt.xlabel("Category")
plt.ylabel("Quantity")
plt.xticks(range(1,len(ct_dis.index)+1),ct_dis.index)
plt.grid()
autolabel(rects)
plt.savefig("./ct_dis_fig")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB() 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_train[:,0],y_test[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_set[:,0],y_set[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_set[:,[0,1,2]],y_set[:,3])
plt.title('Naive Bayes Visualization')
plt.show()    
"""
Created on Mon Apr 23 21:59:43 2018

@author: neera
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_set[:,[0,1,2]],y_set[:,3])
plt.title('Naive Bayes Visualization')
plt.show()    

plt.scatter(X_set[y_set == 0,0], X_set[y_set == 0,1],s=25,c='red',label='cluster 1')
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

#plt.scatter(X_set[:,[0,1,2]],y_set[:,3])
plt.title('Naive Bayes Visualization')
plt.show()    

plt.scatter(X_set[y_set == 0,0], X_set[y_set == 0,1],s=25,c='red',label='cluster 1')
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_set[:,0],y_set[:,3])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_set[:,0],y_set[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))
# visualize the training data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))
from matplotlib.colors import ListedColormap
X#_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
    A=X_train
    B=y_train
plt.scatter(A[:,0],B[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X#_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
A=X_train
B=y_train
plt.scatter(A[:,0],B[:,1])
plt.title('Naive Bayes Visualization')
plt.show()   
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green')),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green')))


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j],X_set[y_set==j],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j],X_set[y_set==j],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j],X_set[y_set==j,3],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j],X_set[y_set==j],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,1],X_set[y_set==j,0],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j,[0,1,2]],X_set[y_set==j,[3]],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j],X_set[y_set==j],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    

from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==0],X_set[y_set==1],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()  
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set],X_set[y_set],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==[0,1,2]],X_set[y_set==[3]],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
    plt.scatter(X_set[y_set==j],X_set[y_set==j],
                c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_set[:,0],y_train[:,1])
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j],X_set[y_set==j],
    #            c=ListedColormap(('red','green'))(i),label=j)

#plt.scatter(X_set[:,0],y_train[:,1])
plt.bar(np.arange(0,4))
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j],X_set[y_set==j],
    #            c=ListedColormap(('red','green'))(i),label=j)

#plt.scatter(X_set[:,0],y_train[:,1])
plt.bar(np.arange(0,4),height=20)
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j],X_set[y_set==j],
    #            c=ListedColormap(('red','green'))(i),label=j)

#plt.scatter(X_set[:,0],y_train[:,1])
plt.bar(np.arange(0,3),height=20)
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

#for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   # plt.scatter(X_set[y_set==j],X_set[y_set==j],
    #            c=ListedColormap(('red','green'))(i),label=j)

#plt.scatter(X_set[:,0],y_train[:,1])
plt.bar(np.arange(0,3),height=20)
plt.xticks(np.arange(0,3),height=15)
plt.title('Naive Bayes Visualization')
plt.show()    
from matplotlib.colors import ListedColormap
X_set,y_set =X_train,y_train

for i,j in enumerate(np.unique(y_set)):
    #plt.scatter(X_train[:,0],y_train[:,1])
   plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)


#plt.scatter(X_train[:,0],y_test[:,1])
plt.title('Naive Bayes Visualization')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pylab import *
import matplotlib.pyplot as plt

df=pd.read_csv("classification_problem_Full.csv")

print(df.head(10))
print("CATE",df["CATEGORY*"].value_counts())

#CATEGORY
ct_dis=df["CATEGORY*"].value_counts()
figure(2)
rects=plt.bar(range(1,len(ct_dis.index)+1),ct_dis.values)
plt.title("Category DISTRIBUTE")
plt.xlabel("Category")
plt.ylabel("Quantity")
plt.xticks(range(1,len(ct_dis.index)+1),ct_dis.index)
plt.grid()
autolabel(rects)
plt.savefig("./ct_dis_fig")

g = sns.factorplot(x="PURPOSE*", y="MILES*", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
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

g = sns.factorplot(x="PURPOSE*", y="MILES*", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
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
runfile('C:/Users/neera/.spyder-py3/NaiveBayes.py', wdir='C:/Users/neera/.spyder-py3')
"""
Created on Thu Apr 19 00:31:27 2018

@author: neera
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

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

g = sns.factorplot(x="PURPOSE*", y="Gender", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
"""
Created on Thu Apr 19 00:31:27 2018

@author: neera
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

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

mygraph = sns.factorplot(x="PURPOSE*", y="Gender", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
mygraph = sns.factorplot(x="PURPOSE*", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")

GendervsAge = sns.factorplot(x="Gender*", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")

CategoryvsAge = sns.factorplot(x="CATEGORY*", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
GendervsAge = sns.factorplot(x="Gender", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
CategoryvsAge = sns.factorplot(x="CATEGORY*", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="PURPOSE*", data=df,
                   size=10, kind="bar", palette="muted")

GendervsAge = sns.factorplot(x="Gender", y="Age", hue="Gender", data=df,
                   size=10, kind="bar", palette="muted")

CategoryvsAge = sns.factorplot(x="CATEGORY*", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="CATEGORY*", data=df,
                   size=10, kind="bar", palette="muted")
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="PURPOSE*", data=df,
                   size=10, kind="bar", palette="muted")
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="PURPOSE*", data=mydata,
                   size=10, kind="bar", palette="muted")
GendervsAge = sns.factorplot(x="Gender", y="Age", hue="Gender", data=mydata,
                   size=10, kind="bar", palette="muted")
CategoryvsAge = sns.factorplot(x="CATEGORY*", y="Age", hue="CATEGORY*", data=mydata,
                   size=10, kind="bar", palette="muted")
GendervsAge = sns.factorplot(x="Gender", y="Age", hue="Gender", data=mydata,
                   size=10, kind="bar", palette="muted")
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="PURPOSE*", data=mydata,
                   size=10, kind="bar", palette="muted")
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

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

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
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="CATEGORY*", data=mydata,
                   size=10, kind="bar", palette="muted")
PurposevsAge = sns.factorplot(x="PURPOSE*", y="Age", hue="PURPOSE*", data=mydata,
                   size=10, kind="bar", palette="muted")
import pandas as pd

a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_PR8.csv')

merge=a.merge(b)
import pandas as pd

a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_PR8.csv')

merge=a.merge(b)
print(merge)
import pandas as pd

a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_PR8.csv')

merge=a.merge(b)
merge.to_csv('Result.csv')
print(merge)
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['PURPOSE*','Fare'])

pd.merge([Dataa,Datab],how='outer')
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

pd.merge([Dataa,Datab],on='CATEGORY*',how='outer')
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

pd.merge([Dataa,Datab],on='CATEGORY*',how='right')
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

#pd.merge([Dataa,Datab],on='CATEGORY*',how='right')

merged=Dataa.merge(Datab)
print(merged)
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

Dataa
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

Dataa
#pd.merge([Dataa,Datab],on='CATEGORY*',how='right')

merged=Dataa.merge(Datab)
merged
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

Dataa
DataR=pd.merge(left=Dataa,right=Datab,how='right',left_on='CATEGORY*',right_on'CATEGORY*')
DataR
#merged=Dataa.merge(Datab)
#merged
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

Dataa
DataR=pd.merge(left=Dataa,right=Datab,how='right',left_on='CATEGORY*', right_on='CATEGORY*')
DataR
#merged=Dataa.merge(Datab)
#merged
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

Dataa
DataR=pd.merge(left=Dataa,right=Datab,how='outer',left_on='CATEGORY*', right_on='CATEGORY*')
DataR
#merged=Dataa.merge(Datab)
#merged
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

Dataa
DataR=pd.merge(left=Dataa,right=Datab,how='outer')
DataR
#merged=Dataa.merge(Datab)
#merged
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

frames=[Dataa,Datab]

result=pd.concat(frames)
result
import pandas as pd

#a=pd.read_csv('classification_problem_Full.csv')
#b=pd.read_csv('DM_PR8.csv')
#
#merge=a.merge(b)
#merge.to_csv('Result.csv')
#print(merge)


a=pd.read_csv('classification_problem_Full.csv')
b=pd.read_csv('DM_prj_Final.csv')

a

Dataa=pd.DataFrame(a,columns=['CATEGORY*','Gender','Age'])
Datab=pd.DataFrame(b,columns=['CATEGORY*','PURPOSE*','Fare'])

frames=[Dataa,Datab]

result=pd.concat(frames)
result
result.to_csv('Result.csv')
#Dataa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier(n_neighbors=3) 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier(n_neighbors=3) 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
mydata=pd.read_csv('classification_problem.csv')

mydata.hist()
pyplot.show()
mydata.replace('?','?',-99999,inplace=True) 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance') 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)

#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
#accuracy_training=classifier.score(X_train,y_train)
#accuracy_testing=classifier.score(X_test,y_test)
#print(accuracy_training)
#print(accuracy_testing)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,1,1]])
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
#print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB() 
classifier.fit(X_train,y_train)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
classifier=tree.DecisionTreeClassifier(criterion='gini')
clf=classifier.fit(X_train,y_train)
#classifier.predict([[1,1,1]])
import graphviz
dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
graph.render("Decision tree")
conda update conda

## ---(Tue Apr 24 23:52:20 2018)---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)













X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area


classifier=tree.DecisionTreeClassifier(criterion='gini')
clf=classifier.fit(X_train,y_train)
#classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
y_pred=classifier.predict([[0,0,0]])
print(y_pred)
import graphviz
dot_data=tree.export_graphviz(clf,out_file=None)
graph=graphviz.Source(dot_data)
graph.render("Decision tree")
y_pred=classifier.predict([[0,0,0]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import svm
classifier=svm.SVC() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import svm
classifier=svm.SVC() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,1,1]])
print(y_pred)
y_pred=classifier.predict([[0,0,0]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)













X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import tree
classifier=tree.DecisionTreeClassifier(criterion='gini')
clf=classifier.fit(X_train,y_train)
#classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,0,0]])
print(y_pred)
y_pred=classifier.predict([[0,1,0]])
print(y_pred)
y_pred=classifier.predict([[0,1,1]])
print(y_pred)
y_pred=classifier.predict([[1,0,0]])
print(y_pred)
runfile('C:/Users/neera/.spyder-py3/temp.py', wdir='C:/Users/neera/.spyder-py3')
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
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
@author: neera
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# Run until finding best number of clusters then run below code

# applying hierarchical clustering to our dataset 
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
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
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# Run until finding best number of clusters then run below code

# applying hierarchical clustering to our dataset 
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
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
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
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
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
#X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns
X=mydata.iloc[:,[10,11,12]].values 
mydata=pd.read_csv('DM_PR8.csv')
#X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns
X=mydata.iloc[:,[10,11,12]].values 

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# Run until finding best number of clusters then run below code

# applying hierarchical clustering to our dataset 
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')

plt.title('Hierarchical clustering')
plt.xlabel('Category')
plt.ylabel('Age')
plt.legend()
plt.show()
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')

plt.title('Hierarchical clustering')
#plt.xlabel('Category')
#plt.ylabel('Age')
plt.legend()
plt.show()
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
#X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns
X=mydata.iloc[:,[10,11,12]].values 

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# Run until finding best number of clusters then run below code

# applying hierarchical clustering to our dataset 
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)


# visualization of our hierarchical clustering
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')

plt.title('Hierarchical clustering')
#plt.xlabel('Category')
#plt.ylabel('Age')
plt.legend()
plt.show()
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')

plt.title('Hierarchical clustering')
#plt.xlabel('Category')
#plt.ylabel('Age')
plt.legend()
plt.show()
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)


# visualization of our hierarchical clustering
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')

plt.title('Hierarchical clustering')
#plt.xlabel('Category')
#plt.ylabel('Age')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = KFold(n_splits=10,shuffle=False,random_state=None)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = KFold(n_splits=10,shuffle=False,random_state=None)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = KFold(n_splits=10)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
X_train, X_test, y_train, y_test = kFold(n_splits=10)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

#K fold cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for train, test in kf.split(mydata):

"""
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1=pd.read_csv('Uberdrives1.csv')
uber1
uber1=pd.read_csv('Uberdrives1.csv')
uber1
uber1.replace('?',-99999,inplace=True) 
uber1.to_csv(uberdrives1.csv’ encoding=’utf-8’)
uber1.to_csv('uberdrives1.csv’ encoding=’utf-8’)
uber1.to_csv('uberdrives1.csv’)
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1
uber1.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value

uber1.to_csv('uberdrives1.csv’)
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1
uber1.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
uber1
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1
uber1.replace(' ',-99999,inplace=True)   # finding the missing data and replacing it by a value
uber1
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1
uber1.replace('',-99999,inplace=True)   # finding the missing data and replacing it by a value
uber1
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1.PURPOSE=uber1.PURPOSE.fillna('')
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1.head()
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1.head()
uber1.PURPOSE=uber1.PURPOSE.fillna('')
uber1
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1.head()
uber1.PURPOSE=uber1.PURPOSE.fillna('9999')
uber1
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv')
uber1.head()
uber1.PURPOSE=uber1.PURPOSE.fillna('9999')
uber1.PURPOSE
"""
Created on Tue Apr 24 09:49:55 2018

@author: neera
"""
import pandas as pd


# getting uber drives file 1

uber1=pd.read_csv('Uberdrives1.csv') # loading data 1
uber1.head()                # printing initial rows of data
uber1.PURPOSE=uber1.PURPOSE.fillna('9999') # Replace missing data in PURPOSE column data with value '9999'
uber1.PURPOSE
uber1.head()                # printing initial rows of data
uber1.START_DATE=uber1.START_DATE.fillna('9999') # Replace missing data in START_DATE column data with value '9999'
uber1.START_DATE
uber1.END_DATE=uber1.END_DATE.fillna('9999') # Replace missing data in END_DATE column data with value '9999'
uber1.END_DATE
uber1.head()                # printing initial rows of data
uber1.MILES=uber1.MILES.fillna('9999') # Replace missing data in MILES column data with value '9999'
uber1.MILES
uber1.head()                # printing initial rows of data
uber1.CATEGORY=uber1.CATEGORY.fillna('9999') # Replace missing data in CATEGORY column data with value '9999'
uber1.CATEGORY
uber1.head()                # printing initial rows of data
uber1.START=uber1.START.fillna('9999') # Replace missing data in START column data with value '9999'
uber1.START



# finding missing values in STOP 
uber1.head()                # printing initial rows of data
uber1.STOP=uber1.STOP.fillna('9999') # Replace missing data in STOP column data with value '9999'
uber1.STOP
uber1.head()                # printing initial rows of data
uber1.START=uber1.START.fillna('9999') # Replace missing data in START column data with value '9999'
uber1.START
uber1.head()                # printing initial rows of data
uber1.START=uber1.START.fillna('9999') # Replace missing data in START column data with value '9999'
uber1.START
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv') # loading data 1
uber1.info()
uber1['PURPOSE'].unique()
uber1.info()
uber1['PURPOSE'].unique()
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True)
uber1.head() 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv') # loading data 1
uber1.head() 
uber1.head()  
uber1=pd.DataFrame.from_csv('Uberdrives1.csv') # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()  
uber1.loc[:,'START_DATE'] = uber1['START_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m %d %Y %H : %M'))
uber1['DRIVE_TIME']=uber1['END_DATE'] - uber1['START_DATE']
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv') # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()         
uber1['DRIVE_TIME']=uber1['END_DATE'] - uber1['START_DATE']
uber1[-5:]
uber1=uber1[:-1]
uber1
uber1.loc[:,'START_DATE'] = uber1['START_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m %d %Y %H : %M'))
display=sns.factorplot(x="PURPOSE",y="MILES",hue="CATEGORY",data=uber1,size=10,kind="bar",palette="muted")
uber1[,START_DATE:=as.POSIXct(START_DATE,format='%m/%d/%Y %H:%M'),]
uber1[START_DATE:=as.POSIXct(START_DATE,format='%m/%d/%Y %H:%M'),]
uber1[START_DATE=as.POSIXct(START_DATE,format='%m/%d/%Y %H:%M'),]
from datetime import datetime
datetime.strptime(START_DATE,'%m-%d-%Y-%H:%M')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv') # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()                # printing initial rows of data
from datetime import datetime
datetime.strptime(START_DATE,'%m-%d-%Y-%H:%M')
uber1.MILES
MILES
uber1.MILES
uber1.MILES*
uber1.MILES
MILES=uber1.MILES
MILES
MILES=uber1[:,5]
MILES
MILES=uber1[:,5]
miles=uber1.iloc[:,5].values
Miles
miles=uber1.iloc[:,5].values
miles
uber1
uber1.info()
miles=uber1.iloc[:,4].values
miles
uber1=pd.DataFrame.from_csv('Uberdrives1.csv') # loading data 1
uber1.head() 
uber1.info()   
miles=uber1.iloc[:,4].values
miles
from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(1,10)
minmax.fit(miles).transform(miles)
miles=uber1.iloc[:,4].values
miles
from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(1,10)
miles=uber1.iloc[:,4].values
miles
from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(1,10))
minmax.fit(miles).transform(miles)
miles=uber1.iloc[:,4].values
miles
from sklearn import preprocessing
minmax=preprocessing.MinMaxScaler(feature_range=(1,10))
minmax.fit(miles).transform(miles)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv') # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()                # printing initial rows of data
uber1.info() 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()                # printing initial rows of data
uber1.head()                # printing initial rows of data
uber1.START_DATE=uber1.START_DATE.fillna('9999') # Replace missing data in START_DATE column data with value '9999'
uber1.START_DATE
from datetime import datetime
uber1.loc[:,'START_DATE'] = uber1['START_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m/%d/%Y %H:%M'))
START_DATE
uber1.START_DATE
from datetime import datetime
uber1.loc[:,'START_DATE'] = uber1['START_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m/%d/%Y %H:%M'))
uber1.START_DATE
from datetime import datetime
uber1.loc[:,'START_DATE'] = uber1['START_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m/%d/%Y %H:%M'))
uber1.info()
uber1.loc[:,'END_DATE'] = uber1['END_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m/%d/%Y %H:%M'))
uber1.info()
uber1.to_csv('uberdrives1')
uber1.to_csv('uberdrives1.csv')
uber1.info()
uber1.to_csv('uberdrives1.csv')
uber1.to_csv('C:\Users\neera\Desktop\Data warehouse\Final_Project\uber1.csv')
uber1.to_csv('uberdrives1.csv')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

#K fold cross validation
#from sklearn.model_selection import KFold
#kf = KFold(n_splits=3)
#for train, test in kf.split(mydata):

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
#X_train, X_test, y_train, y_test = kFold(n_splits=10)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import tree
classifier=tree.DecisionTreeClassifier(criterion='gini')
clf=classifier.fit(X_train,y_train)
#classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,0,0]])
print(y_pred)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)

#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
#accuracy_training=classifier.score(X_train,y_train)
#accuracy_testing=classifier.score(X_test,y_test)
#print(accuracy_training)
#print(accuracy_testing)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)

#accuracy_training= np.sum(classifier.predict(X_train)== y_train) / float(y_train.size)
#accuracy_testing= np.sum(classifier.predict(X_test)== y_test) / float(y_test.size)
#accuracy_training=classifier.score(X_train,y_train)
#accuracy_testing=classifier.score(X_test,y_test)
#print(accuracy_training)
#print(accuracy_testing)


# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)


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

y_pred=classifier.predict(X_test)
#y_pred=classifier.predict([[1,1,1]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)


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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

#K fold cross validation
#from sklearn.model_selection import KFold
#kf = KFold(n_splits=3)
#for train, test in kf.split(mydata):

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
#X_train, X_test, y_train, y_test = kFold(n_splits=10)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import tree
classifier=tree.DecisionTreeClassifier(criterion='gini')
clf=classifier.fit(X_train,y_train)
#classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,0,0]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')

#mydata.hist()
#pyplot.show()
#


#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column

# normalize data
#Nor = mydata.values
#Nor = Nor.reshape((len(Nor), 1))
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler = scaler.fit(Nor)


#discretize age
#pandas.cut(Age, bins, right=True, labels=["Y","M","O"], retbins=False, precision=3, include_lowest=False)

X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

#K fold cross validation
#from sklearn.model_selection import KFold
#kf = KFold(n_splits=3)
#for train, test in kf.split(mydata):

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)
#X_train, X_test, y_train, y_test = kFold(n_splits=10)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import tree
classifier=tree.DecisionTreeClassifier(criterion='gini')
clf=classifier.fit(X_train,y_train)
#classifier.predict([[1,1,1]])

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[0,0,0]])
print(y_pred)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import svm
classifier=svm.SVC() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,0,0])
print(y_pred)
"""
Created on Sun Apr 22 16:17:23 2018

@author: neera
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:31:27 2018

@author: neera
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import svm
classifier=svm.SVC() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,0,0]])
print(y_pred)
"""
Created on Sun Apr 22 16:17:23 2018

@author: neera
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:31:27 2018

@author: neera
"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing our csv dataset

mydata=pd.read_csv('classification_problem_Full.csv')
#mydata.replace('?',-99999,inplace=True)   # finding the missing data and replacing it by a value
#mydata.drop([''],1,inplace=True)   # Dropping unwanted column
X=mydata.iloc[:,[0,1,2]].values
y=mydata.iloc[:,[3]].values

# splitting data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Classification algorithm area

from sklearn import svm
classifier=svm.SVC() 
classifier.fit(X_train,y_train)

# get output of test results

#y_pred=classifier.predict(X_test)
y_pred=classifier.predict([[1,0,0]])
print(y_pred)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

# importing our csv dataset

mydata=pd.read_csv('DM_PR8.csv')
X=mydata.iloc[:,[6,13]].values

# Find out the best number of clusters

Array=[]     # to store sum of squares within the groups

for i in range(1,14):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0) 
    kmeans.fit(X)
    Array.append(kmeans.inertia_)    # inertia --> Sum of squared distances of samples to their closest cluster center


plt.plot(range(1,14),Array)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squares within groups')
plt.show()

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
"""
Created on Sat Apr 21 14:24:56 2018

@author: neera
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

#x=np.array([[15,30],[60,25],[15,78],[10,40]])
mydata=pd.read_csv('DM_PR8.csv')
#X=mydata.iloc[:,[2,10,11,12]].values #  loading category and age columns
X=mydata.iloc[:,[10,11,12]].values 

# finding best number of clusters

import scipy.cluster.hierarchy as sc
dendrogram=sc.dendrogram(sc.linkage(X,method='ward'))

# Run until finding best number of clusters then run below code

# applying hierarchical clustering to our dataset 
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)


# visualization of our hierarchical clustering
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')

plt.title('Hierarchical clustering')
#plt.xlabel('Category')
#plt.ylabel('Age')
plt.legend()
plt.show()
from sklearn.cluster import AgglomerativeClustering
Hierarchical=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
PredictH=Hierarchical.fit_predict(X)
plt.scatter(X[PredictH==0,0],X[PredictH==0,1],s=500,c='red',label='cluster 1')
plt.scatter(X[PredictH==1,0],X[PredictH==1,1],s=500,c='blue',label='cluster 2')
plt.scatter(X[PredictH==2,0],X[PredictH==2,1],s=500,c='green',label='cluster 3')
uber1=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2

uber1=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.info()
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.info()
uber2.head()
uber2.describe()
uber2['Gender'].unique()
uber2.head()
uber2.head(100)
uber2.head(50,100)
uber2.head(60)
uber1['Status'].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2['Status'].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2['status'].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2['Status'].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown'
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2

uber2.info()

uber2.head(60)   # printing first 60 columns of data
uber2.describe()
uber2['Status'].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2['Status '].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2.head(60)
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.head(60) 
uber2.info()
uber2.Ride Duration(mins)
duration=uber2.Ride Duration(mins)
duration=uber2.Ride Duration.(mins)
duration=uber2.Ride Duration
duration=uber2.Ride.Duration
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2

uber2.info()
duration=uber2.Ride Duration 
duration=uber2.Ride.Duration 
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2

uber2.info()
duration=uber2.Ride_Duration
duration
duration.fillna(method='ffill')  
duration.fillna(method='ffill')  
uber2.head(60)
duration.fillna(method='ffill') 
uber2.head(60) 
duration.fillna(method='ffill') 
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.isnull()
uber2.isnull().sum() 
import sys
print (sys.platform)
print (2 ** 100)
raw_input( )
import sys
print (sys.platform)
print (2 ** 100)
rawinput( )
import sys
print (sys.platform)
print (2 ** 100)
input( )
import sys
#
fare=input("Enter the fare recorded")
fare=fare.upper()
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()
uber2.isnull().sum() 
uber2.head(60)   # printing first 60 columns of data
uber2.describe()
uber2['Status '].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2.head(60) 
uber2.to_csv('uber2test.csv',index=False)
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()
uber2.head(60)
uber2.describe()
uber2['Status '].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2.head(60) 
duration=uber2.Ride_Duration
duration
duration.fillna(method='ffill') 
uber2.head(60) 
duration=uber2.Ride_Duration
duration

duration.fillna(method='ffill')   
uber2.to_csv('uber2test.csv',index=False)
duration=uber2.Ride_Duration
duration

duration.fillna(method='ffill')  
uber2.head(60) 
duration=uber2.Ride_Duration
duration

duration.fillna(method='ffill') 
duration=uber2['Ride_Duration ']
duration=uber2['Ride_Duration']
duration.fillna(method='ffill') 
uber2.to_csv('uber2test.csv',index=False)
uber2['Ride_Duration'].fillna(method='ffill') 
uber2.to_csv('uber2test.csv',index=False)
uber2['Ride_Duration'].fillna(np.nan,method='ffill',inplace=True)  
uber2['Ride_Duration'].fillna(np.nan,method='ffill')  
uber2['Ride_Duration'].fillna(method='ffill')  
uber2['Days'].replace('sunday,saturday','WEEKEND',inplace=True)
uber2.head(60) 
uber2['Days'].replace('saturday','WEEKEND',inplace=True)
uber2.head(60) 
uber2['Days'].replace('Saturday','WEEKEND',inplace=True)
uber2.head(60) 
uber2['Days'].replace('Saturday','WEEKEND',inplace=True)
uber2['Days'].replace('Sunday','WEEKEND',inplace=True)
uber2.head(60) 
uber2['Days'].replace('Monday','WEEKDAY',inplace=True)
uber2['Days'].replace('Tuesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Wednesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Thursday','WEEKDAY',inplace=True)
uber2['Days'].replace('Friday','WEEKDAY',inplace=True)
uber2.head(60) 
uber2.to_csv('uber2test.csv',index=False)
uber1.merge(uber2)
uber1.merge(uber2),to_csv('Check.csv')
uber1.merge(uber2).to_csv('Check.csv')

## ---(Sat May  5 11:32:22 2018)---
uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()      
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()         # Summary of our data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.isnull()   # return if null or not
uber1.isnull().sum() 
uber1.info()     
uber1.isnull()   
uber1.isnull().sum() 
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()
uber2.isnull().sum() 
uber2.head(60)
uber2['Ride_Duration'].fillna(method='ffill') 
uber2['Days'].replace('Saturday','WEEKEND',inplace=True)
uber2['Days'].replace('Sunday','WEEKEND',inplace=True)
uber2.head(60) 

# replacing days column by WEEKDAY value
uber2['Days'].replace('Monday','WEEKDAY',inplace=True)
uber2['Days'].replace('Tuesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Wednesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Thursday','WEEKDAY',inplace=True)
uber2['Days'].replace('Friday','WEEKDAY',inplace=True)
uber2.head(60) 
uber2.to_csv('uber2test.csv',index=False)
uber2.head(60) 
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()

uber2.head(60)   # printing first 60 columns of data
uber2.describe()
uber2['Status '].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 
uber2.to_csv('uber2test.csv',index=False)
uber2.isnull().sum() 
uber2.to_csv('uber2test.csv',index=False)
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()

uber2.head(60)   # printing first 60 columns of data
uber2.describe()
uber2['Status '].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 

uber2.head(60) 

duration=uber2['Ride_Duration']
duration=uber2.Ride_Duration
duration

uber2['Ride_Duration'].fillna(method='ffill')     # By forward fill, filling the missing value in ride duration feature


# replacing days column by WEEKEND values

uber2['Days'].replace('Saturday','WEEKEND',inplace=True)
uber2['Days'].replace('Sunday','WEEKEND',inplace=True)
uber2.head(60) 

# replacing days column by WEEKDAY value
uber2['Days'].replace('Monday','WEEKDAY',inplace=True)
uber2['Days'].replace('Tuesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Wednesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Thursday','WEEKDAY',inplace=True)
uber2['Days'].replace('Friday','WEEKDAY',inplace=True)
uber2.head(60) 
uber2.to_csv('uber2test.csv',index=False)
uber1.merge(uber2).to_csv('Result.csv')
display=sns.factorplot(x="PURPOSE",y="MILES",hue="CATEGORY",data=Result,size=10,kind="bar",palette="muted")
UBERFINAL=pd.DataFrame.from_csv('Result.csv',index_col=None) 
display=sns.factorplot(x="PURPOSE",y="MILES",hue="CATEGORY",data=UBERFINAL,size=10,kind="bar",palette="muted")
UBERFINAL.grouby(['CATEGORY'])['MILES'].sum()/df['MILES'].sum()
UBERFINAL.groupby(['CATEGORY'])['MILES'].sum()/df['MILES'].sum()
UBERFINAL.groupby(['CATEGORY'])['MILES'].sum()/UBERFINAL['MILES'].sum()
Startlocation=UBERFINAL["START"].value_counts()
Startlocation=UBERFINAL["START"].value_counts()
Startlocation.sort_values(inplace=True,ascending=False)
Startlocation=Startlocation.iloc[:3]
print("START LOCATION FOR UBER DRIVE \n",Startlocation)
Endlocation=UBERFINAL["START"].value_counts()
Endlocation.sort_values(inplace=True,ascending=False)
Endlocation=Endlocation.iloc[:4]
print("END LOCATION COUNT FOR UBER DRIVE \n",Endlocation)
MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for MILES in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">=25"]+=1
        
        MILES_COUNT=pd.series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)

MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">=25"]+=1
        
        MILES_COUNT=pd.series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)

MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)

MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.Series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)

MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.Series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)
        
        plotme=plt.bar(range(1,len(MILES_COUNT)+1),MILES_COUNT.values)
        plt.title("MILES")
        plt.xlabel("MILES")
        plt.ylabel("QUANTITY")
        plt.xticks(range(1,len(MILES_COUNT)+1),MILES_COUNT.index)
        plt.grid()
        autolabel(plotme)
        plt.savefig("./MILES_COUNT_figure")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.Series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)
        
        plotme=plt.bar(range(1,len(MILES_COUNT)+1),MILES_COUNT.values)
        plt.title("MILES")
        plt.xlabel("MILES")
        plt.ylabel("QUANTITY")
        plt.xticks(range(1,len(MILES_COUNT)+1),MILES_COUNT.index)
        plt.grid()
        autolabel(plotme)
        plt.savefig("./MILES_COUNT_figure")

PURPOSE_COUNT=UBERFINAL["PURPOSE"].value_counts()
PURPOSE_COUNT.sort_values(ascending=False)
PURPOSE_COUNT=PURPOSE_COUNT.iloc[:6]
print("PURPOSE COUNT: \n",PURPOSE_COUNT)   
CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
autolabel(plot2)
plt.savefig("./category_figure")
START_COUNT=UBERFINAL["START"].value_counts()
plot3=plt.bar(range(1,len(START_COUNT.index)+1),START_COUNT.values)
plt.title("START LOCATION DISTRIBUTION")
plt.xlabel("START LOCATION")
plt.ylabel("COUNT")
plt.xticks(range(1,len(START_COUNT.index)+1),START_COUNT.index)
plt.grid()
autolabel(plot3)
plt.savefig("./StartLocation_figure")
count=UBERFINAL['PURPOSE'].value_counts().tolist()
purpose=UBERFINAL['PURPOSE'].value_counts().index.tolist()
purposevscount=list(zip(purpose,count))
purposevscount=pd.DataFrame(purposevscount,columns=['PURPOSE','COUNT'])

plot3=sns.barplot(x='COUNT',y='PURPOSE',data=purposevscount,order=purposevscount['PURPOSE'].tolist())
plot3.set(xlabel='Number of Rides',ylabel='purpose')
plt.show()
UBERFINAL['CATEGORY'].value_counts()

CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
autolabel(plot2)
plt.savefig("./category_figure")
UBERFINAL['CATEGORY'].value_counts()

CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
plt.savefig("./category_figure")
g=nx.Graph()

g=nx.from_pandas_dataframe(rides,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(g))
import networkx as nx
g=nx.Graph()

g=nx.from_pandas_dataframe(rides,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(g))
import networkx as nx
g=nx.Graph()

g=nx.from_pandas_dataframe(UBERFINAL,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(g))
import networkx as nx
g=nx.Graph()

g=nx.from_pandas_dataframe(UBERFINAL,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(g))

plt.figure(figsize=(12,12))
nx.draw_circular(g,with_labels=True,node_size=100)
plt.show()
import networkx as nx
plot=nx.Graph()

plot=nx.from_pandas_dataframe(UBERFINAL,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(plot))

plt.figure(figsize=(12,12))
nx.draw_circular(plot,with_labels=True,node_size=100)
plt.show()
plot4<-ggplot(UBERFINAL,aes(x=PURPOSE,fill=PURPOSE))+geom_bar(aes(y=(..count..)/sum(..count..)))+ scale_y_continuous(labels = scales::percent)+ylab("Percentage")+coord_flip()+ggtitle("Uber Trips for Various Purposes")
plot4=ggplot(UBERFINAL,aes(x=PURPOSE,fill=PURPOSE))+geom_bar(aes(y=(..count..)/sum(..count..)))+ scale_y_continuous(labels = scales::percent)+ylab("Percentage")+coord_flip()+ggtitle("Uber Trips for Various Purposes")
plot5=
Jan = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 1]
Feb = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 2]
Mar = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 3]
Apr = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 4]
May = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 5]
Jun = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 6]
Jul = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 7]
Aug = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 8]
Sep = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 9]
Oct = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 10]
Nov = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 11]
Dec = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 12]
Jan = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 1]
Feb = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 2]
Mar = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 3]
Apr = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 4]
May = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 5]
Jun = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 6]
Jul = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 7]
Aug = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 8]
Sep = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 9]
Oct = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 10]
Nov = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 11]
Dec = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 12]

Jan.loc[:,'day'] = pd.to_datetime(Jan['START_DATE']).dt.day
Feb.loc[:,'day'] = pd.to_datetime(Feb['START_DATE']).dt.day
Mar.loc[:,'day'] = pd.to_datetime(Mar['START_DATE']).dt.day
Apr.loc[:,'day'] = pd.to_datetime(Apr['START_DATE']).dt.day
May.loc[:,'day'] = pd.to_datetime(May['START_DATE']).dt.day
Jun.loc[:,'day'] = pd.to_datetime(Jun['START_DATE']).dt.day
Jul.loc[:,'day'] = pd.to_datetime(Jul['START_DATE']).dt.day
Aug.loc[:,'day'] = pd.to_datetime(Aug['START_DATE']).dt.day
Sep.loc[:,'day'] = pd.to_datetime(Sep['START_DATE']).dt.day
Oct.loc[:,'day'] = pd.to_datetime(Oct['START_DATE']).dt.day
Nov.loc[:,'day'] = pd.to_datetime(Nov['START_DATE']).dt.day
Dec.loc[:,'day'] = pd.to_datetime(Dec['START_DATE']).dt.day

Jan_group = Jan.groupby(['day']).agg('sum')
Feb_group = Feb.groupby(['day']).agg('sum')
Mar_group = Mar.groupby(['day']).agg('sum')
Apr_group = Apr.groupby(['day']).agg('sum')
May_group = May.groupby(['day']).agg('sum')
Jun_group = Jun.groupby(['day']).agg('sum')
Jul_group = Jul.groupby(['day']).agg('sum')
Aug_group = Aug.groupby(['day']).agg('sum')
Sep_group = Sep.groupby(['day']).agg('sum')
Oct_group = Oct.groupby(['day']).agg('sum')
Nov_group = Nov.groupby(['day']).agg('sum')
Dec_group = Dec.groupby(['day']).agg('sum')
miles_day_frame = pd.concat([Jan_group, Feb_group,Mar_group,Apr_group,May_group,Jun_group,Jul_group,Aug_group,Sep_group,Oct_group,
               Nov_group,Dec_group],ignore_index=True, axis=1)
miles_day_frame.columns = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
miles_day_frame.fillna(0,inplace=True)
plt.show()
Jan_group.plot()
Dec_group.plot()
plt.figure(figsize=(10,10))
UBERFINAL['PURPOSE'].value_counts()[:6].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0])
plt.show()
plt.figure(figsize=(10,10))
UBERFINAL['PURPOSE'].value_counts()[:6].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0])
plt.show()
plt.figure(figsize=(10,10))
UBERFINAL['PURPOSE'].value_counts()[:6].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0])
plt.show()
plt.figure(figsize=(10,10))
UBERFINAL['PURPOSE'].value_counts()[:6].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0])
plt.show()
x=np.arange[0,1156]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
miles_day_frame = pd.concat([Jan_group, Feb_group,Mar_group,Apr_group,May_group,Jun_group,Jul_group,Aug_group,Sep_group,Oct_group,
               Nov_group,Dec_group],ignore_index=True, axis=1)
miles_day_frame.columns = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
miles_day_frame.fillna(0,inplace=True)

plt.show()
x=np.arange[0,1156]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
import numpy as np
x=np.arange[0,1156]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
x=np.arange[0,1156]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
x=np.arange[0,1156]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
import numpy as np
x=np.arange[0,1156]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
x=np.arange[0,1155]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
import numpy as np
x=np.arange[0,1155]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
x=np.arange[0,1155]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()

x=np.arange[0,1155]
y=UBERFINAL['MILES']
plt.figure(figsize=(18,8))

plt.scatter(x,y,s=15)
plt.xticks([0,400,800,1200])
plt.show()
g=sns.FacetGrid(data=UBERFINAL,aspect=2,size=8,hue='PURPOSE')
g.map(plt.plot(),'START_DATE')
plt.legend()
plt.xlabel('Number of Trips')
plt.show()
g=sns.FacetGrid(data=UBERFINAL,aspect=2,size=8,hue='PURPOSE')
g.map(plt.plot,'START_DATE')
plt.legend()
plt.xlabel('Number of Trips')
plt.show()
uber1.loc[:,'START_DATE'] = uber1['START_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m/%d/%Y %H:%M'))
g=sns.FacetGrid(data=UBERFINAL,aspect=2,size=8,hue='PURPOSE')
g.map(plt.plot,'START_DATE')
plt.legend()
plt.xlabel('Number of Trips')
plt.show()
g=sns.FacetGrid(data=UBERFINAL,aspect=2,size=8,hue='PURPOSE')
g.map(plt.plot,'CATEGORY')
plt.legend()
plt.xlabel('Number of Trips')
plt.show()
Jan_group.plot()
Dec_group.plot()
g=sns.FacetGrid(data=UBERFINAL,aspect=2,size=8,hue='PURPOSE')
g.map(plt.plot,'MILES')
plt.legend()
plt.xlabel('Number of Trips')
plt.show()
g=sns.FacetGrid(data=UBERFINAL,aspect=2,size=8,hue='PURPOSE')
g.map(plt.plot,'START_DATE')
plt.legend()
plt.xlabel('Number of Trips')
plt.show()
g=sns.FacetGrid(data=UBERFINAL,aspect=2,size=8,hue='PURPOSE')
g.map(plt.plot,pd.to_datetime(UBERFINAL['START_DATE']))
plt.legend()
plt.xlabel('Number of Trips')
plt.show()
plt.figure(figsize=(18,10))
UBERFINAL['START'].value_counts()[:50].plot(kind='bar')
plt.show()
plt.figure(figsize=(18,10))
UBERFINAL['STOP'].value_counts()[:50].plot(kind='bar')
plt.show()
totals = UBERFINAL.groupby('CATEGORY*', as_index=False).agg({'MILES*': 'sum'})
totals['PERCENTAGE'] = (totals['MILES*']/UBERFINAL['MILES*'].sum())*100
totals
totals = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totals['PERCENTAGE'] = (totals['MILES']/UBERFINAL['MILES*'].sum())*100
totals
totals = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totals['PERCENTAGE'] = (totals['MILES']/UBERFINAL['MILES'].sum())*100
totals
totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp
totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp

sizes = np.array(totalp['PERCENTAGE'])
labels = np.array(totalp['CATEGORY*'])


fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizes, explode=[0.2,0,0], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('PERCENTAGE OF MILES BY CATEGORY')

plt.show()
totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp

sizes = np.array(totalp['PERCENTAGE'])
labels = np.array(totalp['CATEGORY'])


fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizes, explode=[0.2,0,0], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('PERCENTAGE OF MILES BY CATEGORY')

plt.show()
totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp

sizes = np.array(totalp['PERCENTAGE'])
labels = np.array(totalp['CATEGORY'])


fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizes, explode=[0.2,0], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('PERCENTAGE OF MILES BY CATEGORY')

plt.show()
totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp

sizes = np.array(totalp['PERCENTAGE'])
labels = np.array(totalp['CATEGORY'])


fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizes, explode=[0.2,0], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('MILES BY CATEGORY IN PERCENTAGE')

plt.show()

totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp

sizearray = np.array(totalp['PERCENTAGE'])
labels = np.array(totalp['CATEGORY'])


fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizearray, explode=[0.2,0], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('MILES BY CATEGORY IN PERCENTAGE')

plt.show()
UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
UBERFINAL['Hour'] = UBERFINAL['START_DATE*'].apply(lambda time: time.hour)
UBERFINAL['Month'] = UBERFINAL['START_DATE*'].apply(lambda time: time.month)
UBERFINAL['Day of Week'] = UBERFINAL['START_DATE*'].apply(lambda time: time.dayofweek)
UBERFINAL['Date'] = UBERFINAL['START_DATE*'].apply(lambda time: time.date())
UBERFINAL.head()
UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
UBERFINAL['Hour'] = UBERFINAL['START_DATE'].apply(lambda time: time.hour)
UBERFINAL['Month'] = UBERFINAL['START_DATE'].apply(lambda time: time.month)
UBERFINAL['Day of Week'] = UBERFINAL['START_DATE'].apply(lambda time: time.dayofweek)
UBERFINAL['Date'] = UBERFINAL['START_DATE'].apply(lambda time: time.date())
UBERFINAL.head()

daymap ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(daymap)
UBERFINAL.head()
Convertme ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(daymap)
UBERFINAL.head()
Convertme ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(Convertme)
UBERFINAL.head()

daymap ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(daymap)
UBERFINAL.head()

UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
UBERFINAL['Hour'] = UBERFINAL['START_DATE'].apply(lambda time: time.hour)
UBERFINAL['Month'] = UBERFINAL['START_DATE'].apply(lambda time: time.month)
UBERFINAL['Day of Week'] = UBERFINAL['START_DATE'].apply(lambda time: time.dayofweek)
UBERFINAL['Date'] = UBERFINAL['START_DATE'].apply(lambda time: time.date())
UBERFINAL.head()



# Convert 'Day of Week' from numerical to text(that we can understand)
daymap ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(daymap)
UBERFINAL.head()
UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
UBERFINAL['Hour'] = UBERFINAL['START_DATE'].apply(lambda time: time.hour)
UBERFINAL['Month'] = UBERFINAL['START_DATE'].apply(lambda time: time.month)
UBERFINAL['Day of Week'] = UBERFINAL['START_DATE'].apply(lambda time: time.dayofweek)
UBERFINAL['Date'] = UBERFINAL['START_DATE'].apply(lambda time: time.date())
UBERFINAL.head()



# Convert 'Day of Week' from numerical to text(that we can understand)
convertme ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(convertme)
UBERFINAL.head()
plt.figure(figsize=(20,8))
sns.countplot(x='Day of Week',data = UBERFINAL,hue = 'PURPOSE*')
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad=0.)
plt.figure(figsize=(20,8))
sns.countplot(x='Day of Week',data = UBERFINAL,hue = 'PURPOSE')
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad=0.)
MilesPerMonth = UBERFINAL.groupby('Month')['MILES'].sum()

plt.figure(figsize=(20,8))
sns.barplot(x='Month',y='MILES*',data=MilesPerMonth.reset_index())
plt.tight_layout()
MilesPerMonth = UBERFINAL.groupby('Month')['MILES'].sum()

plt.figure(figsize=(20,8))
sns.barplot(x='Month',y='MILES',data=MilesPerMonth.reset_index())
plt.tight_layout()
sns.lmplot(x='Month',y='PURPOSE',data=UBERFINAL.groupby('Month').count().reset_index())
df = UBERFINAL[UBERFINAL.variety.isin(UBERFINAL.variety.value_counts().head(5).index)]

sns.boxplot(
    x='MILES',
    y='PURPOSE',
    data=UBERFINAL
)
df = UBERFINAL[UBERFINAL.MILES.isin(UBERFINAL.MILES.value_counts().head(5).index)]

sns.boxplot(
    x='MILES',
    y='PURPOSE',
    data=UBERFINAL
)
df = UBERFINAL[UBERFINAL.PURPOSE.isin(UBERFINAL.PURPOSE.value_counts().head(5).index)]

sns.boxplot(
    x='MILES',
    y='PURPOSE',
    data=UBERFINAL
)

df = UBERFINAL[UBERFINAL.Fare.isin(UBERFINAL.Fare.value_counts().head(5).index)]

sns.boxplot(
    x='MILES',
    y='PURPOSE',
    data=UBERFINAL
)

sns.violinplot(
    x='variety',
    y='points',
    data=UBERFINAL[UBERFINAL.MILES.isin(UBERFINAL.MILES.value_counts()[:5].index)]
)
sns.violinplot(
    x='variety',
    y='points',
    data=UBERFINAL[UBERFINAL.MILES.isin(UBERFINAL.MILES.value_counts()[:5].index)]
)
sns.violinplot(
    x='MILES',
    y='PURPOSE',
    data=UBERFINAL[UBERFINAL.MILES.isin(UBERFINAL.MILES.value_counts()[:5].index)]
)
sns.violinplot(
    x='CATEGORY',
    y='PURPOSE',
    data=UBERFINAL[UBERFINAL.MILES.isin(UBERFINAL.MILES.value_counts()[:5].index)]
)
sns.violinplot(
    x='Fare',
    y='Ride_Duration',
    data=UBERFINAL[UBERFINAL.Ride_Duration.isin(UBERFINAL.Ride_Duration.value_counts()[:5].index)]
)
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Fare'])
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Ride duration'])

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Ride_Duration'])
df = UBERFINAL.head(1000).dropna()

(
    ggplot(UBERFINAL)
        + geom_point()
        + aes(color='points')
        + aes('Ride_Duration', 'Fare')
        + stat_smooth()
)
from ggplot import *
df = UBERFINAL.head(1000).dropna()

(
    ggplot(UBERFINAL)
        + geom_point()
        + aes(color='points')
        + aes('Ride_Duration', 'Fare')
        + stat_smooth()
)
from ggplot import *
(ggplot(UBERFINAL)
     + aes('points', 'variety')
     + geom_bin2d(bins=20)
)
library(ggplot2)
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Ride_Duration'])
miles_day_frame = pd.concat([Jan_group, Feb_group,Mar_group,Apr_group,May_group,Jun_group,Jul_group,Aug_group,Sep_group,Oct_group,
               Nov_group,Dec_group],ignore_index=True, axis=1)
miles_day_frame.columns = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
miles_day_frame.fillna(0,inplace=True)

plt.show()
sns.violinplot(
    x='Fare',
    y='Ride_Duration',
    data=UBERFINAL[UBERFINAL.Ride_Duration.isin(UBERFINAL.Ride_Duration.value_counts()[:5].index)]
)
Startlocation=UBERFINAL["START"].value_counts()
Startlocation.sort_values(inplace=True,ascending=False)
Startlocation=Startlocation.iloc[:3]
print("START LOCATION COUNT FOR UBER DRIVE \n",Startlocation)
PURPOSE_COUNT=UBERFINAL["PURPOSE"].value_counts()
PURPOSE_COUNT.sort_values(ascending=False)
PURPOSE_COUNT=PURPOSE_COUNT.iloc[:6]
print("PURPOSE COUNT: \n",PURPOSE_COUNT)   
UBERFINAL['CATEGORY'].value_counts()

CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
plt.savefig("./category_figure")
START_COUNT=UBERFINAL["START"].value_counts()
plot3=plt.bar(range(1,len(START_COUNT.index)+1),START_COUNT.values)
plt.title("START LOCATION DISTRIBUTION")
plt.xlabel("START LOCATION")
plt.ylabel("COUNT")
plt.xticks(range(1,len(START_COUNT.index)+1),START_COUNT.index)
plt.grid()
autolabel(plot3)
plt.savefig("./StartLocation_figure")
display=sns.factorplot(x="PURPOSE",y="MILES",hue="CATEGORY",data=UBERFINAL,size=10,kind="bar",palette="muted")
display=sns.factorplot(x="PURPOSE",y="Days",hue="CATEGORY",data=UBERFINAL,size=10,kind="bar",palette="muted")
display=sns.factorplot(x="PURPOSE",y="Days",hue="Days",data=UBERFINAL,size=10,kind="bar",palette="muted")
display=sns.factorplot(x="PURPOSE",y="Fare",hue="Fare",data=UBERFINAL,size=10,kind="bar",palette="muted")
display=sns.factorplot(x="PURPOSE",y="Fare",hue="PURPOSE",data=UBERFINAL,size=10,kind="bar",palette="muted")
MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.Series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)
        
        plotme=plt.bar(range(1,len(MILES_COUNT)+1),MILES_COUNT.values)
        plt.title("MILES")
        plt.xlabel("MILES")
        plt.ylabel("QUANTITY")
        plt.xticks(range(1,len(MILES_COUNT)+1),MILES_COUNT.index)
        plt.grid()
        plt.savefig("./MILES_COUNT_figure")

MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.Series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)
        
        plotme=plt.bar(range(1,len(MILES_COUNT)+1),MILES_COUNT.values)
        plt.title("MILES")
        plt.xlabel("MILES")
        plt.ylabel("QUANTITY")
        plt.xticks(range(1,len(MILES_COUNT)+1),MILES_COUNT.index)
        plt.grid()
        plt.savefig("./MILES_COUNT_figure")

count=UBERFINAL['PURPOSE'].value_counts().tolist()
purpose=UBERFINAL['PURPOSE'].value_counts().index.tolist()
purposevscount=list(zip(purpose,count))
purposevscount=pd.DataFrame(purposevscount,columns=['PURPOSE','COUNT'])

plot3=sns.barplot(x='COUNT',y='PURPOSE',data=purposevscount,order=purposevscount['PURPOSE'].tolist())
plot3.set(xlabel='Number of Rides',ylabel='purpose')
plt.show()
plt.figure(figsize=(12,12))
nx.draw_circular(plot,with_labels=True,node_size=100)
plt.show()
import networkx as nx
plot=nx.Graph()

plot=nx.from_pandas_dataframe(UBERFINAL,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(plot))

plt.figure(figsize=(12,12))
nx.draw_circular(plot,with_labels=True,node_size=100)
plt.show()
Jan = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 1]
Feb = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 2]
Mar = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 3]
Apr = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 4]
May = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 5]
Jun = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 6]
Jul = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 7]
Aug = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 8]
Sep = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 9]
Oct = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 10]
Nov = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 11]
Dec = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 12]

Jan.loc[:,'day'] = pd.to_datetime(Jan['START_DATE']).dt.day
Feb.loc[:,'day'] = pd.to_datetime(Feb['START_DATE']).dt.day
Mar.loc[:,'day'] = pd.to_datetime(Mar['START_DATE']).dt.day
Apr.loc[:,'day'] = pd.to_datetime(Apr['START_DATE']).dt.day
May.loc[:,'day'] = pd.to_datetime(May['START_DATE']).dt.day
Jun.loc[:,'day'] = pd.to_datetime(Jun['START_DATE']).dt.day
Jul.loc[:,'day'] = pd.to_datetime(Jul['START_DATE']).dt.day
Aug.loc[:,'day'] = pd.to_datetime(Aug['START_DATE']).dt.day
Sep.loc[:,'day'] = pd.to_datetime(Sep['START_DATE']).dt.day
Oct.loc[:,'day'] = pd.to_datetime(Oct['START_DATE']).dt.day
Nov.loc[:,'day'] = pd.to_datetime(Nov['START_DATE']).dt.day
Dec.loc[:,'day'] = pd.to_datetime(Dec['START_DATE']).dt.day

Jan_group = Jan.groupby(['day']).agg('sum')
Feb_group = Feb.groupby(['day']).agg('sum')
Mar_group = Mar.groupby(['day']).agg('sum')
Apr_group = Apr.groupby(['day']).agg('sum')
May_group = May.groupby(['day']).agg('sum')
Jun_group = Jun.groupby(['day']).agg('sum')
Jul_group = Jul.groupby(['day']).agg('sum')
Aug_group = Aug.groupby(['day']).agg('sum')
Sep_group = Sep.groupby(['day']).agg('sum')
Oct_group = Oct.groupby(['day']).agg('sum')
Nov_group = Nov.groupby(['day']).agg('sum')
Dec_group = Dec.groupby(['day']).agg('sum')
START_COUNT=UBERFINAL["START"].value_counts()
plot3=plt.bar(range(1,len(START_COUNT.index)+1),START_COUNT.values)
plt.title("START LOCATION DISTRIBUTION")
plt.xlabel("START LOCATION")
plt.ylabel("COUNT")
plt.xticks(range(1,len(START_COUNT.index)+1),START_COUNT.index)
plt.grid()
#autolabel(plot3)
plt.savefig("./StartLocation_figure")
count=UBERFINAL['PURPOSE'].value_counts().tolist()
purpose=UBERFINAL['PURPOSE'].value_counts().index.tolist()
purposevscount=list(zip(purpose,count))
purposevscount=pd.DataFrame(purposevscount,columns=['PURPOSE','COUNT'])

plot3=sns.barplot(x='COUNT',y='PURPOSE',data=purposevscount,order=purposevscount['PURPOSE'].tolist())
plot3.set(xlabel='Number of Rides',ylabel='purpose')
plt.show()
CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
plt.savefig("./category_figure")
print(UBERFINAL['CATEGORY'].value_counts())
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Ride_Duration'])
display=sns.factorplot(x="PURPOSE",y="MILES",hue="CATEGORY",data=UBERFINAL,size=10,kind="bar",palette="muted")
UBERFINAL.groupby(['CATEGORY'])['MILES'].sum()/UBERFINAL['MILES'].sum()
Startlocation=UBERFINAL["START"].value_counts()
Startlocation.sort_values(inplace=True,ascending=False)
Startlocation=Startlocation.iloc[:3]
print("START LOCATION COUNT FOR UBER DRIVE \n",Startlocation)
Endlocation=UBERFINAL["START"].value_counts()
Endlocation.sort_values(inplace=True,ascending=False)
Endlocation=Endlocation.iloc[:4]
print("END LOCATION COUNT FOR UBER DRIVE \n",Endlocation)
MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.Series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)
        
        plotme=plt.bar(range(1,len(MILES_COUNT)+1),MILES_COUNT.values)
        plt.title("MILES")
        plt.xlabel("MILES")
        plt.ylabel("QUANTITY")
        plt.xticks(range(1,len(MILES_COUNT)+1),MILES_COUNT.index)
        plt.grid()
        plt.savefig("./MILES_COUNT_figure")

display=sns.factorplot(x="PURPOSE",y="Fare",hue="PURPOSE",data=UBERFINAL,size=10,kind="bar",palette="muted")
PURPOSE_COUNT=UBERFINAL["PURPOSE"].value_counts()
PURPOSE_COUNT.sort_values(ascending=False)
PURPOSE_COUNT=PURPOSE_COUNT.iloc[:6]
print("PURPOSE COUNT: \n",PURPOSE_COUNT)   
print(UBERFINAL['CATEGORY'].value_counts())
print(UBERFINAL['CATEGORY'].value_counts())

CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
plt.savefig("./category_figure")
START_COUNT=UBERFINAL["START"].value_counts()
plot3=plt.bar(range(1,len(START_COUNT.index)+1),START_COUNT.values)
plt.title("START LOCATION DISTRIBUTION")
plt.xlabel("START LOCATION")
plt.ylabel("COUNT")
plt.xticks(range(1,len(START_COUNT.index)+1),START_COUNT.index)
plt.grid()
#autolabel(plot3)
plt.savefig("./StartLocation_figure")
count=UBERFINAL['PURPOSE'].value_counts().tolist()
purpose=UBERFINAL['PURPOSE'].value_counts().index.tolist()
purposevscount=list(zip(purpose,count))
purposevscount=pd.DataFrame(purposevscount,columns=['PURPOSE','COUNT'])

plot3=sns.barplot(x='COUNT',y='PURPOSE',data=purposevscount,order=purposevscount['PURPOSE'].tolist())
plot3.set(xlabel='Number of Rides',ylabel='purpose')
plt.show()
import networkx as nx
plot=nx.Graph()

plot=nx.from_pandas_dataframe(UBERFINAL,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(plot))

plt.figure(figsize=(12,12))
nx.draw_circular(plot,with_labels=True,node_size=100)
plt.show()
Dec_group.plot()
Jan_group.plot()
plt.figure(figsize=(10,10))
UBERFINAL['PURPOSE'].value_counts()[:6].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0])
plt.show()
plt.figure(figsize=(18,10))
UBERFINAL['START'].value_counts()[:50].plot(kind='bar')
plt.show()
plt.figure(figsize=(18,10))
UBERFINAL['STOP'].value_counts()[:50].plot(kind='bar')
plt.show()
totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp

sizearray = np.array(totalp['PERCENTAGE'])
labels = np.array(totalp['CATEGORY'])


fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizearray, explode=[0.2,0], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('MILES BY CATEGORY IN PERCENTAGE')

plt.show()
UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
UBERFINAL['Hour'] = UBERFINAL['START_DATE'].apply(lambda time: time.hour)
UBERFINAL['Month'] = UBERFINAL['START_DATE'].apply(lambda time: time.month)
UBERFINAL['Day of Week'] = UBERFINAL['START_DATE'].apply(lambda time: time.dayofweek)
UBERFINAL['Date'] = UBERFINAL['START_DATE'].apply(lambda time: time.date())
UBERFINAL.head()
UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
UBERFINAL['Hour'] = UBERFINAL['START_DATE'].apply(lambda time: time.hour)
UBERFINAL['Month'] = UBERFINAL['START_DATE'].apply(lambda time: time.month)
UBERFINAL['Day of Week'] = UBERFINAL['START_DATE'].apply(lambda time: time.dayofweek)
UBERFINAL['Date'] = UBERFINAL['START_DATE'].apply(lambda time: time.date())
UBERFINAL.head()



# Converting 'Day of Week' from numerical to categorical or nominal value
convertme ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(convertme)
UBERFINAL.head()
plt.figure(figsize=(20,8))
sns.countplot(x='Day of Week',data = UBERFINAL,hue = 'PURPOSE')
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad=0.)
MilesPerMonth = UBERFINAL.groupby('Month')['MILES'].sum()

plt.figure(figsize=(20,8))
sns.barplot(x='Month',y='MILES',data=MilesPerMonth.reset_index())
plt.tight_layout()
sns.lmplot(x='Month',y='PURPOSE',data=UBERFINAL.groupby('Month').count().reset_index())
miles_day_frame = pd.concat([Jan_group, Feb_group,Mar_group,Apr_group,May_group,Jun_group,Jul_group,Aug_group,Sep_group,Oct_group,
               Nov_group,Dec_group],ignore_index=True, axis=1)
miles_day_frame.columns = ['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
miles_day_frame.fillna(0,inplace=True)

plt.show()
sns.violinplot(
    x='Fare',
    y='Ride_Duration',
    data=UBERFINAL[UBERFINAL.Ride_Duration.isin(UBERFINAL.Ride_Duration.value_counts()[:5].index)]
)
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Ride_Duration'])
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()                # printing initial rows of data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()                # printing initial rows of data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()  
uber1.head() 
uber1.head() 
uber2.to_csv('uber2test.csv',index=False)
uber2.head(60) 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()                # printing initial rows of data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# getting uber drives file 1

uber1=pd.DataFrame.from_csv('Uberdrives1.csv',index_col=None) # loading data 1
uber1.head() 
uber1.info()                     # Summary of our data
uber1['PURPOSE'].unique()     #getting the unique values of purpose
uber1['PURPOSE'].replace(np.nan,'Other', inplace=True) #Replacing missing value by 'Other' 
uber1.head()                # printing initial rows of data
uber1.head()                # printing initial rows of data
uber1.START_DATE=uber1.START_DATE.fillna('9999') # Replace missing data in START_DATE column data with value '9999'
uber1.START_DATE
uber1.END_DATE=uber1.END_DATE.fillna('9999') # Replace missing data in END_DATE column data with value '9999'
uber1.END_DATE
uber1.head()                # printing initial rows of data
uber1.CATEGORY=uber1.CATEGORY.fillna('9999') # Replace missing data in CATEGORY column data with value '9999'
uber1.CATEGORY



# finding missing values in START 
uber1.head()                # printing initial rows of data
uber1.START=uber1.START.fillna('9999') # Replace missing data in START column data with value '9999'
uber1.START


# finding missing values in STOP 
uber1.head()                # printing initial rows of data
uber1.STOP=uber1.STOP.fillna('9999') # Replace missing data in STOP column data with value '9999'
uber1.STOP
uber1.info()
uber2=pd.DataFrame.from_csv('Uberdrives2.csv',index_col=None) # loading data 2
uber2.isnull()   # return if null or not
uber2.isnull().sum() 
uber2.info()
uber2.head(60)   # printing first 60 columns of data
uber2.describe()
uber2['Status '].replace(np.nan,'UNKNOWN', inplace=True)  #Replacing missing value by 'Unknown' 

uber2.head(60) 
uber2['Ride_Duration'].fillna(method='ffill')  
uber2['Days'].replace('Saturday','WEEKEND',inplace=True)
uber2['Days'].replace('Sunday','WEEKEND',inplace=True)
uber2.head(60) 
uber2['Days'].replace('Monday','WEEKDAY',inplace=True)
uber2['Days'].replace('Tuesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Wednesday','WEEKDAY',inplace=True)
uber2['Days'].replace('Thursday','WEEKDAY',inplace=True)
uber2['Days'].replace('Friday','WEEKDAY',inplace=True)
uber2.head(60) 
uber2.head(60) 
display=sns.factorplot(x="PURPOSE",y="MILES",hue="CATEGORY",data=UBERFINAL,size=10,kind="bar",palette="muted")

UBERFINAL.groupby(['CATEGORY'])['MILES'].sum()/UBERFINAL['MILES'].sum()
# 94% of miles were due to the contribution of business trips
Startlocation=UBERFINAL["START"].value_counts()
Startlocation.sort_values(inplace=True,ascending=False)
Startlocation=Startlocation.iloc[:3]
print("START LOCATION COUNT FOR UBER DRIVE \n",Startlocation)

# End location of drive
Endlocation=UBERFINAL["START"].value_counts()
Endlocation.sort_values(inplace=True,ascending=False)
Endlocation=Endlocation.iloc[:4]
print("END LOCATION COUNT FOR UBER DRIVE \n",Endlocation)
MILES_COUNT=UBERFINAL["MILES"]
MILES_RANGE=["<=5","5-10","10-15","15-20","20-25",">25"]
MILES_DICT=dict()
for item in MILES_RANGE:
    MILES_DICT[item]=0

for mile in MILES_COUNT.values:
    if mile<=5:
        MILES_DICT["<=5"]+=1
    elif mile<=10:
        MILES_DICT["5-10"]+=1
    elif mile<=15:
        MILES_DICT["10-15"]+=1
    elif mile<=20:
        MILES_DICT["15-20"]+=1
    elif mile<=25:
        MILES_DICT["20-25"]+=1
    else:
        MILES_DICT[">25"]+=1
        
        MILES_COUNT=pd.Series(MILES_DICT)
        MILES_COUNT.sort_values(inplace=True,ascending=False)
        print("Miles:\n",MILES_COUNT)
        
        plotme=plt.bar(range(1,len(MILES_COUNT)+1),MILES_COUNT.values)
        plt.title("MILES")
        plt.xlabel("MILES")
        plt.ylabel("QUANTITY")
        plt.xticks(range(1,len(MILES_COUNT)+1),MILES_COUNT.index)
        plt.grid()
        plt.savefig("./MILES_COUNT_figure")

display=sns.factorplot(x="PURPOSE",y="Fare",hue="PURPOSE",data=UBERFINAL,size=10,kind="bar",palette="muted")

# purpose count
PURPOSE_COUNT=UBERFINAL["PURPOSE"].value_counts()
PURPOSE_COUNT.sort_values(ascending=False)
PURPOSE_COUNT=PURPOSE_COUNT.iloc[:6]
print("PURPOSE COUNT: \n",PURPOSE_COUNT)   
print(UBERFINAL['CATEGORY'].value_counts())

CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
plt.savefig("./category_figure")
START_COUNT=UBERFINAL["START"].value_counts()
plot3=plt.bar(range(1,len(START_COUNT.index)+1),START_COUNT.values)
plt.title("START LOCATION DISTRIBUTION")
plt.xlabel("START LOCATION")
plt.ylabel("COUNT")
plt.xticks(range(1,len(START_COUNT.index)+1),START_COUNT.index)
plt.grid()
#autolabel(plot3)
plt.savefig("./StartLocation_figure")
count=UBERFINAL['PURPOSE'].value_counts().tolist()
purpose=UBERFINAL['PURPOSE'].value_counts().index.tolist()
purposevscount=list(zip(purpose,count))
purposevscount=pd.DataFrame(purposevscount,columns=['PURPOSE','COUNT'])

plot3=sns.barplot(x='COUNT',y='PURPOSE',data=purposevscount,order=purposevscount['PURPOSE'].tolist())
plot3.set(xlabel='Number of Rides',ylabel='purpose')
plt.show()
import networkx as nx
plot=nx.Graph()

plot=nx.from_pandas_dataframe(UBERFINAL,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(plot))

plt.figure(figsize=(12,12))
nx.draw_circular(plot,with_labels=True,node_size=100)
plt.show()
Jan = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 1]
Feb = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 2]
Mar = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 3]
Apr = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 4]
May = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 5]
Jun = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 6]
Jul = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 7]
Aug = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 8]
Sep = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 9]
Oct = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 10]
Nov = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 11]
Dec = UBERFINAL[pd.to_datetime(UBERFINAL['START_DATE']).dt.month == 12]

Jan.loc[:,'day'] = pd.to_datetime(Jan['START_DATE']).dt.day
Feb.loc[:,'day'] = pd.to_datetime(Feb['START_DATE']).dt.day
Mar.loc[:,'day'] = pd.to_datetime(Mar['START_DATE']).dt.day
Apr.loc[:,'day'] = pd.to_datetime(Apr['START_DATE']).dt.day
May.loc[:,'day'] = pd.to_datetime(May['START_DATE']).dt.day
Jun.loc[:,'day'] = pd.to_datetime(Jun['START_DATE']).dt.day
Jul.loc[:,'day'] = pd.to_datetime(Jul['START_DATE']).dt.day
Aug.loc[:,'day'] = pd.to_datetime(Aug['START_DATE']).dt.day
Sep.loc[:,'day'] = pd.to_datetime(Sep['START_DATE']).dt.day
Oct.loc[:,'day'] = pd.to_datetime(Oct['START_DATE']).dt.day
Nov.loc[:,'day'] = pd.to_datetime(Nov['START_DATE']).dt.day
Dec.loc[:,'day'] = pd.to_datetime(Dec['START_DATE']).dt.day

Jan_group = Jan.groupby(['day']).agg('sum')
Feb_group = Feb.groupby(['day']).agg('sum')
Mar_group = Mar.groupby(['day']).agg('sum')
Apr_group = Apr.groupby(['day']).agg('sum')
May_group = May.groupby(['day']).agg('sum')
Jun_group = Jun.groupby(['day']).agg('sum')
Jul_group = Jul.groupby(['day']).agg('sum')
Aug_group = Aug.groupby(['day']).agg('sum')
Sep_group = Sep.groupby(['day']).agg('sum')
Oct_group = Oct.groupby(['day']).agg('sum')
Nov_group = Nov.groupby(['day']).agg('sum')
Dec_group = Dec.groupby(['day']).agg('sum')
Jan_group.plot()
Dec_group.plot()
plt.figure(figsize=(10,10))
UBERFINAL['PURPOSE'].value_counts()[:6].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0])
plt.show()
plt.figure(figsize=(18,10))
UBERFINAL['START'].value_counts()[:50].plot(kind='bar')
plt.show()
plt.figure(figsize=(18,10))
UBERFINAL['STOP'].value_counts()[:50].plot(kind='bar')
plt.show()
totalp = UBERFINAL.groupby('CATEGORY', as_index=False).agg({'MILES': 'sum'})
totalp['PERCENTAGE'] = (totalp['MILES']/UBERFINAL['MILES'].sum())*100
totalp

sizearray = np.array(totalp['PERCENTAGE'])
labels = np.array(totalp['CATEGORY'])


fig1, ax1 = plt.subplots(figsize=(9,9))
ax1.pie(sizearray, explode=[0.2,0], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('MILES BY CATEGORY IN PERCENTAGE')

plt.show()
UBERFINAL['START_DATE'] = pd.to_datetime(UBERFINAL['START_DATE'])
UBERFINAL['END_DATE'] = pd.to_datetime(UBERFINAL['END_DATE'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
UBERFINAL['Hour'] = UBERFINAL['START_DATE'].apply(lambda time: time.hour)
UBERFINAL['Month'] = UBERFINAL['START_DATE'].apply(lambda time: time.month)
UBERFINAL['Day of Week'] = UBERFINAL['START_DATE'].apply(lambda time: time.dayofweek)
UBERFINAL['Date'] = UBERFINAL['START_DATE'].apply(lambda time: time.date())
UBERFINAL.head()
convertme ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
UBERFINAL['Day of Week'] = UBERFINAL['Day of Week'].map(convertme)
UBERFINAL.head()
plt.figure(figsize=(20,8))
sns.countplot(x='Day of Week',data = UBERFINAL,hue = 'PURPOSE')
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad=0.)
MilesPerMonth = UBERFINAL.groupby('Month')['MILES'].sum()

plt.figure(figsize=(20,8))
sns.barplot(x='Month',y='MILES',data=MilesPerMonth.reset_index())
plt.tight_layout()
sns.lmplot(x='Month',y='PURPOSE',data=UBERFINAL.groupby('Month').count().reset_index())
sns.violinplot(
    x='Fare',
    y='Ride_Duration',
    data=UBERFINAL[UBERFINAL.Ride_Duration.isin(UBERFINAL.Ride_Duration.value_counts()[:5].index)]
)

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Ride_Duration'])