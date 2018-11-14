# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 09:49:55 2018

@author: neera
"""
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



# finding missing values in start_date 
uber1.head()                # printing initial rows of data
uber1.START_DATE=uber1.START_DATE.fillna('9999') # Replace missing data in START_DATE column data with value '9999'
uber1.START_DATE

# finding missing values in end_date 
uber1.END_DATE=uber1.END_DATE.fillna('9999') # Replace missing data in END_DATE column data with value '9999'
uber1.END_DATE


# finding missing values in MILES 
uber1.head()                # printing initial rows of data
uber1.MILES=uber1.MILES.fillna('9999') # Replace missing data in MILES column data with value '9999'
uber1.MILES


# finding missing values in CATEGORY 
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


#converting data type of start_date from String to DateTime
from datetime import datetime
uber1.loc[:,'START_DATE'] = uber1['START_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m/%d/%Y %H:%M'))

#converting data type of End_date from String to DateTime
uber1.loc[:,'END_DATE'] = uber1['END_DATE'].apply(lambda x:pd.datetime.strptime(x,'%m/%d/%Y %H:%M'))

uber1.info()

# saving the modified file
uber1.to_csv('uberdrives1.csv')

# loading data 2

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
uber2.head(60) 


#COMBINING OR MERGING TWO CSV FILES
uber1.merge(uber2).to_csv('Result.csv')

UBERFINAL=pd.DataFrame.from_csv('Result.csv',index_col=None) 

# graph displays main contributors for miles and also their purpose
display=sns.factorplot(x="PURPOSE",y="MILES",hue="CATEGORY",data=UBERFINAL,size=10,kind="bar",palette="muted")



# percentage of business miles vs personal miles 
UBERFINAL.groupby(['CATEGORY'])['MILES'].sum()/UBERFINAL['MILES'].sum()
# 94% of miles were due to the contribution of business trips


# Start Location of drive
Startlocation=UBERFINAL["START"].value_counts()
Startlocation.sort_values(inplace=True,ascending=False)
Startlocation=Startlocation.iloc[:3]
print("START LOCATION COUNT FOR UBER DRIVE \n",Startlocation)

# End location of drive
Endlocation=UBERFINAL["START"].value_counts()
Endlocation.sort_values(inplace=True,ascending=False)
Endlocation=Endlocation.iloc[:4]
print("END LOCATION COUNT FOR UBER DRIVE \n",Endlocation)

#MILES
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
        
# short distances are most preferred observed from this data        

# graph displays main contributors for purpose and also their fare
display=sns.factorplot(x="PURPOSE",y="Fare",hue="PURPOSE",data=UBERFINAL,size=10,kind="bar",palette="muted")

# purpose count
        
PURPOSE_COUNT=UBERFINAL["PURPOSE"].value_counts()
PURPOSE_COUNT.sort_values(ascending=False)
PURPOSE_COUNT=PURPOSE_COUNT.iloc[:6]
print("PURPOSE COUNT: \n",PURPOSE_COUNT)   

# maximum count is for Other which is 503 and the next highest is for meeting followed by Meal/Entertainment


# Category count
print(UBERFINAL['CATEGORY'].value_counts())

CATEGORY_COUNT=UBERFINAL["CATEGORY"].value_counts()
plot2=plt.bar(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.values)
plt.title("CATEGORY DISTRIBUTION")
plt.xlabel("CATEGORY")
plt.ylabel("COUNT")
plt.xticks(range(1,len(CATEGORY_COUNT.index)+1),CATEGORY_COUNT.index)
plt.grid()
plt.savefig("./category_figure")


# Start location plot

START_COUNT=UBERFINAL["START"].value_counts()
plot3=plt.bar(range(1,len(START_COUNT.index)+1),START_COUNT.values)
plt.title("START LOCATION DISTRIBUTION")
plt.xlabel("START LOCATION")
plt.ylabel("COUNT")
plt.xticks(range(1,len(START_COUNT.index)+1),START_COUNT.index)
plt.grid()
#autolabel(plot3)
plt.savefig("./StartLocation_figure")

# plot for type of purpose and number of rides 


count=UBERFINAL['PURPOSE'].value_counts().tolist()
purpose=UBERFINAL['PURPOSE'].value_counts().index.tolist()
purposevscount=list(zip(purpose,count))
purposevscount=pd.DataFrame(purposevscount,columns=['PURPOSE','COUNT'])

plot3=sns.barplot(x='COUNT',y='PURPOSE',data=purposevscount,order=purposevscount['PURPOSE'].tolist())
plot3.set(xlabel='Number of Rides',ylabel='purpose')
plt.show()


#graph
import networkx as nx
plot=nx.Graph()

plot=nx.from_pandas_dataframe(UBERFINAL,source='START',target='STOP',edge_attr=['START_DATE','END_DATE','CATEGORY','MILES','PURPOSE'])

print(nx.info(plot))

plt.figure(figsize=(12,12))
nx.draw_circular(plot,with_labels=True,node_size=100)
plt.show()


# Miles vs Category

#plot4=ggplot(UBERFINAL,aes(x=PURPOSE,fill=PURPOSE))+geom_bar(aes(y=(..count..)/sum(..count..)))+scale_y_continous(labels= scales::percent) +ylab("percentage")+coord_flip()+ggtitle("Uber trips for various purpose")

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

# piechart for purpose

plt.figure(figsize=(10,10))
UBERFINAL['PURPOSE'].value_counts()[:6].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0])
plt.show()


#start location bar chart
plt.figure(figsize=(18,10))
UBERFINAL['START'].value_counts()[:50].plot(kind='bar')
plt.show()

#Stop location bar chart
plt.figure(figsize=(18,10))
UBERFINAL['STOP'].value_counts()[:50].plot(kind='bar')
plt.show()

# calculating percentage of miles by category.
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

##
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



# finding the hidden relationship between the missing value and 'Day of Week'
plt.figure(figsize=(20,8))
sns.countplot(x='Day of Week',data = UBERFINAL,hue = 'PURPOSE')
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad=0.)

##Miles by month

MilesPerMonth = UBERFINAL.groupby('Month')['MILES'].sum()

plt.figure(figsize=(20,8))
sns.barplot(x='Month',y='MILES',data=MilesPerMonth.reset_index())
plt.tight_layout()


# month purpose regression

sns.lmplot(x='Month',y='PURPOSE',data=UBERFINAL.groupby('Month').count().reset_index())


# violin plot 

#A violinplot cleverly replaces the box in the boxplot with a kernel density estimate for the data. It shows basically the same data, but is harder to misinterpret and much prettier than the utilitarian boxplot.


sns.violinplot(
    x='Fare',
    y='Ride_Duration',
    data=UBERFINAL[UBERFINAL.Ride_Duration.isin(UBERFINAL.Ride_Duration.value_counts()[:5].index)]
)




#auto corelation plot

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(UBERFINAL['Ride_Duration'])




