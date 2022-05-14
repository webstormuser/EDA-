#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[123]:


#loading the dataset 
train_df=pd.read_excel('Data_Train.xlsx')
train_df.head()


# In[124]:


test_df=pd.read_excel('Test_set.xlsx')
test_df.head()


# In[125]:


df=train_df.append(test_df)
df.head(10)


# In[126]:


df.info()


# In[127]:


df.columns


# In[128]:


#trying to correct Date_of Journey column
df['Date']=df['Date_of_Journey'].str.split('/').str[0]


# In[129]:


df['Month']=df['Date_of_Journey'].str.split('/').str[1]


# In[130]:


df['Year']=df['Date_of_Journey'].str.split('/').str[2]


# In[131]:


df.head()


# In[132]:


df.info()


# In[133]:


#Since date we have splited but it is in object form so we have to convert into int
df['Date']=df['Date'].astype(int)
df['Month']=df['Month'].astype(int)
df['Year']=df['Year'].astype(int)


# In[134]:


df.drop('Date_of_Journey',axis=1,inplace=True)


# In[135]:


df.head()


# In[136]:


df['Arrival_Time']=df['Arrival_Time'].str.split(' ').str[0]
df['Arrival_Time'].head()


# In[137]:


df['Arrival_Hour']=df['Arrival_Time'].str.split(':').str[0]


# In[138]:


df['Arrival_Minutes']=df['Arrival_Time'].str.split(':').str[1]
df['Arrival_Minutes']


# In[139]:


df['Arrival_Hour']=df['Arrival_Hour'].astype(int)
df['Arrival_Minutes']=df['Arrival_Minutes'].astype(int)


# In[140]:


df.drop(columns=['Arrival_Time'],axis=1,inplace=True)


# In[141]:


df.info()


# In[142]:


df.head()


# In[143]:


df['Dep_Hours']=df['Dep_Time'].str.split(':').str[0]
df['Dep_Hours']=df['Dep_Hours'].astype(int)


# In[144]:


df['Dep_Minutes']=df['Dep_Time'].str.split(':').str[1]
df['Dep_Minutes']=df['Dep_Minutes'].astype(int)


# In[145]:


df.drop('Dep_Time',axis=1,inplace=True)


# In[146]:


df.head()


# In[147]:


df['Total_Stops'].unique()


# In[148]:


df['Total_Stops'].isnull().sum()
#since single record is with nan so we can drop it 


# In[149]:


df.dropna(inplace=True)


# In[150]:


df['Total_Stops'].unique()


# In[151]:


#Handling categorical features in Total stop column
df['Total_Stops']=df['Total_Stops'].map({'non-stop':1,'1 stop':2,'2 stops':3,'3 stops':4})


# In[152]:


df.head()


# In[153]:


#same information is represented in Route column as in no of stops so we can drop column
df.drop('Route',axis=1,inplace=True)


# In[154]:


df.head()


# In[155]:


#Since duration is also object type we have to manage it into numeic format 
df['Duration_Hour']=df['Duration'].str.split(' ').str[0].str.split('h').str[0]
df['Duration_Hour'].unique()


# In[156]:


#one anomalies is in  Duration_Hour value i.e 5m 
df[df['Duration_Hour']=='5m']


# In[157]:


#Above record seams to be suspicious because Mumbai to hyderabad it takes only 5m duration and total stops are 3 so we have to this record


# In[158]:


df.drop(6474,axis=0,inplace=True)


# In[159]:


df.head()


# In[160]:


df['Duration_Minutes']=df['Duration'].str.split(' ').str[1].str.split('m').str[0]


# In[161]:


df['Duration_Minutes']=df['Duration_Minutes'].fillna(0)


# In[162]:


df['Duration_Minutes'].unique()


# In[163]:


df['Duration_Minutes']=df['Duration_Minutes'].astype(int)


# In[164]:


df.drop('Duration',axis=1,inplace=True)


# In[165]:


df.head()


# In[166]:


df['Airline'].unique()


# In[167]:


df['Source'].unique()


# In[168]:


from sklearn.preprocessing import  LabelEncoder
label_encoder=LabelEncoder()
df['Airline']=label_encoder.fit_transform(df['Airline'])


# In[169]:


df['Airline'].unique()


# In[170]:


df['Additional_Info']=label_encoder.fit_transform(df['Additional_Info'])


# In[171]:


df.head()


# In[173]:


df['Source']=label_encoder.fit_transform(df['Source'])


# In[174]:


df.head()


# In[191]:


destination_dummies=pd.get_dummies(df,columns=['Destination'],drop_first=True)


# In[192]:


df2=df.copy()


# In[196]:


df2=destination_dummies


# In[197]:


df2.head()


# In[199]:


df2.info()

