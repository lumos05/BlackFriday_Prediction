#!/usr/bin/env python
# coding: utf-8

# In[414]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[415]:


##importing dataset
df_train = pd.read_csv('BlackFriday_train.csv')
df_train.head()


# In[416]:


df_test = pd.read_csv('BlackFriday_test.csv')
df_test.head()


# In[417]:


#merge train and test data
df=df_train.append(df_test)
df.head()


# In[418]:


df.info()


# In[419]:


df.describe()


# In[420]:


#droping userid coln
df.drop(['User_ID'],axis=1,inplace=True)
df.head()


# In[421]:


#preprocessing 
#Handling gender feature which is categorical
df.Gender.unique()


# In[422]:


df['Gender'] = df['Gender'].map({'F':0,'M':1})
df.head()


# In[423]:


#Handling age cateorical feature
df['Age'].unique()


# In[424]:


#target encoding age column
df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[425]:


df.head()


# In[426]:


#handling city category column
df['City_Category'] = df['City_Category'].map({'A':0,'B':1,'C':2})


# In[ ]:





# In[ ]:





# In[427]:


#dropping City_Category - 2 categories sufficient to reperesent all 3 categories A,B,C
df.head()


# In[428]:


## Checking for missing values
df.isnull().sum()


# In[429]:


# Product_Category_2 and Product_Category_3 have missing values - perform data
#exploration to replace them 

df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[430]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[431]:


df.head()


# In[432]:


df.info()


# In[433]:


## all coloumns have non-null values.
## converting objects,unit8 to integers
df['Stay_In_Current_City_Years'].unique()
#df['B']=df['B'].astype(int)
#df['C']=df['C'].astype(int)
#df.info()


# In[434]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')
df.head()


# In[435]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)


# In[ ]:





# In[436]:


df.info()


# In[437]:


#visualizing the dataset - graph for Age vs Purchase
sns.barplot('Age','Purchase',hue='Gender',data=df)


# In[438]:


## we observe from the graph that purchsing  of men is high compared to women
sns.barplot('Occupation','Purchase',hue='Gender',data=df)


# In[439]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df)


# In[440]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df)


# In[441]:


sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df)


# In[442]:


df.head()


# In[443]:


# feature scaling
df_test=df[df['Purchase'].isnull()]
df_train=df[~df['Purchase'].isnull()]
X=df_train.drop('Purchase',axis=1)


# In[444]:


X=X.drop('Product_ID',axis=1)


# In[445]:


X.head()


# In[446]:


y=df_train['Purchase']


# In[447]:


y.head()


# In[448]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)


# In[449]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[450]:


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from math import *

model = Ridge()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# finding the mean_squared error
mse = mean_squared_error(y_test, y_pred)
print("RMSE Error:", np.sqrt(mse))

# finding the r2 score or the variance
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)


# In[ ]:




