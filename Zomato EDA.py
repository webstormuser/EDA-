#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessart libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


import matplotlib


# In[3]:


#loading dataset 
df=pd.read_excel('zomato.xlsx')


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


#Finding any missing values availble or not
df.isnull().sum()


# In[9]:


#one more method to find null values from column 
[features for features in df.columns if df[features].isnull().sum()>0]


# In[10]:


contry_codes=pd.read_excel('Country-Code.xlsx')


# In[11]:


contry_codes


# In[12]:


contry_codes.columns


# In[13]:


#We have to merge two files using pd.merge 
data=pd.merge(df,contry_codes,how='left',on='Country_Code') #on is used to combine two columns based on key 


# In[14]:


data.head()


# In[15]:


data.columns


# In[16]:


#Finding the restaurant counts for each country 
data['Country'].value_counts()


# In[19]:


#Finding highest coverage of zomato restaurants in a Country .From plot we get India is highest covering country.
matplotlib.rcParams['figure.figsize']=(10,5)
plt.pie(labels=data['Country'].value_counts().head(5).index,x=data['Country'].value_counts().head(5).values,autopct='%1.2f%%',radius=1.5)


# In[20]:


ratings=data.groupby(['Aggregate_rating','Rating_color','Rating_text']).size().reset_index().rename(columns={0:'Rating_Count'})


# In[21]:


ratings


# In[22]:


#figure size 
plt.rcParams['figure.figsize']=(10,5)
sns.barplot(data=ratings,x=ratings['Aggregate_rating'],y=ratings['Rating_Count'],hue=ratings['Rating_color'],palette=['blue','red','orange','yellow','green','darkgreen'])


# In[23]:


df.columns


# In[24]:


zero_rating_data=data[data.Rating_color=='White'].groupby('Country').size().reset_index()


# In[25]:


data.groupby(['Aggregate_rating','Country']).size().reset_index().head(5)#another method for above question 


# In[26]:


data[['Country','Currency']].groupby(['Country','Currency']).size().reset_index()


# In[27]:


online_option=data[data['Has_Online_delivery']=='Yes']


# In[28]:


online_option.groupby(['Country','Has_Online_delivery']).size()


# In[29]:


plt.pie(x=online_option.groupby(['Country','Has_Online_delivery']).size().values,labels=online_option.groupby(['Country','Has_Online_delivery']).size().index,autopct='%1.2f%%')


# In[30]:


#Finding ratio between Has_table_booking to no table booking
len(data[data['Has_Table_booking']=='Yes'])/len(data['Restaurant_ID'])*100


# In[31]:


labels = list(data.City.value_counts().head(10).index)
labels 


# In[32]:


values = list(data.City.value_counts().head(10).values)
values


# In[33]:


#Plotting bar graph to show top 10 cities with highest no of restaurants counts 
plt.figure(figsize=(12,7))
plt.title('Top 10 cities with highest zomato restaurants ')
plt.xlabel('City')
plt.ylabel('Total Restaurant_Count')
sns.barplot(x=labels,y=values)


# # PLotting the visuals for online delivery and online book table for restaurants 

# In[34]:


plt.subplot(1,2,1)
plt.title('Online Delivery status count of Restaurnats')
sns.countplot(data['Has_Online_delivery'])
plt.subplot(1,2,2)
plt.title('Online table booking report for restaurants ')
sns.countplot(data['Has_Table_booking'])


# Is there a difference in no. of votes for the restaurants that deliver and the restaurant that don’t?

# In[35]:


sns.distplot(data['Aggregate_rating'],kde=True,bins=20)


# # Now we will explore the dataset for India 

# In[36]:


data.columns


# In[37]:


#Since some columns are not necessary for our analysis so we can drop it 
data.drop(columns=['Locality','Locality Verbose','Longitude','Latitude','Currency','Address'],inplace=True)


# In[38]:


data.drop(columns=['Country_Code'],inplace=True)


# In[39]:


data.head()


# In[40]:


india_df=data[data['Country']=='India'].reset_index()


# In[41]:


india_df.head()


# In[43]:


#finding the overall relation towards agg_rating ,avg_cost_for_two with Online booking and online delivery in India 
plt.figure(figsize=(12,7))
sns.scatterplot(x=india_df['Aggregate_rating'],y=india_df['Average_Cost_for_two'],hue=data['Has_Online_delivery'])


# In[44]:


india_df['Has_Online_delivery'].value_counts()


# In[45]:


india_df['Has_Table_booking'].value_counts()


# In[46]:


india_df['Is delivering now'].value_counts()


# In[47]:


#From graph it clearly says most of the restaurants lies between 2.5 to 4.5 stars of their rating for restatunat .


# In[48]:


plt.figure(figsize=(12,7))
sns.scatterplot(x=india_df['Aggregate_rating'],y=india_df['Average_Cost_for_two'],hue=data['Has_Table_booking'])


# In[51]:


#Grouping India's dataset citiwise
india_city_df=india_df.groupby('City').mean().reset_index()
india_city_df.head()
#plotting the graph for city Vurses Average cost for two people 
plt.figure(figsize=(15,15))
sns.barplot(x='Average_Cost_for_two',y='City',data=india_city_df.sort_values('Average_Cost_for_two'))
plt.savefig('E:/Data Analyst CapStone/Zomato Project/image9.png')


# In[52]:


#Finding the restaurants and the no of locations they have outlet for restaurant
resta_india_df=india_df['Restaurant_Name'].value_counts().reset_index().rename(columns={'index':'Restaurant_Name','Restaurant_Name':'Count'})
resta_india_df.head()


# In[53]:


resta_india_df


# In[54]:


plt.xticks(rotation=9)
sns.barplot(x='Restaurant_Name',y='Count',data=resta_india_df.head(10))


# What is the maximum and minimum no. of cuisines that a restaurant serves? Also, what is the relationship between No. of cuisines served and Ratings

# In[56]:


#First we have to split the cuisines  and store into separate column
def no_of_Cuisinis(cuisines):
    return len(cuisines.split())


# In[57]:


india_df['No_of_Cuisine_offered']=india_df['Cuisines'].apply(no_of_Cuisinis)


# In[58]:


india_df.head()


# In[59]:


rest_avg_df=india_df.groupby('Restaurant_Name').mean().reset_index()
rest_avg_df


# In[60]:


rest_avg_df.head(10)


# In[61]:


rest_cuisine_df=pd.merge(rest_avg_df,resta_india_df, on='Restaurant_Name')
rest_cuisine_df


# In[62]:


rest_cuisine_df.sort_values(by='Count',ascending=False,inplace=True,ignore_index=True)
rest_cuisine_df.head()


# In[63]:


#from above it clears that some franchise also provide maximum no of cuisines and some single restaurnat also provide more cuisnines 


# In[64]:


#Finding the no of cusiines provided by restaurant 
plt.xlabel('No of Cuisines')
plt.ylabel('No of Restaurant')
sns.histplot(data=india_df,x='No_of_Cuisine_offered')


# In[65]:


india_df.groupby(['No_of_Cuisine_offered','Restaurant_Name']).size().reset_index().sort_values(by='No_of_Cuisine_offered',ascending=False)


# In[66]:


#12 is highest no of cuisine surved by Bikanerwala


# In[62]:


#Potting the visual for no of cuisine offered and rating what they affect 
sns.scatterplot(x='No_of_Cuisine_offered',y='Votes',data=india_df)


# In[67]:


#Checking  how  affects on avg rating  by no of cuisine offered
sns.scatterplot(y='No_of_Cuisine_offered' ,x='Aggregate_rating',data=india_df)


# In[68]:


sns.scatterplot(x='Average_Cost_for_two',y='No_of_Cuisine_offered',data=india_df,hue='Has_Table_booking')


# In[69]:


sns.scatterplot(x='Average_Cost_for_two',y='No_of_Cuisine_offered',data=india_df,hue='Has_Online_delivery')


# In[70]:


#From above visual it clearly indicate that no of cuisines offered by restaurant,Average cost and Online delivery and Online pre table booking affects on average rating


# In[71]:


india_df.drop_duplicates()


# In[72]:


# A function that takes a dataframe as input and returns a new dataframe with the cuisines and number of restaurants it is offerd in, from the input dataframe.

def num_of_restaurants_per_cuisine(df):
    cuisine_count = {}
    for cuisine in df['Cuisines']:
        for c in str(cuisine).split(', '):
            if c in cuisine_count:
                cuisine_count[c] = cuisine_count[c] + 1
            else:
                cuisine_count[c] = 1
    return pd.DataFrame(cuisine_count, index=['count']).transpose().reset_index().rename(columns={'index':'cuisine'})


# In[73]:


cuisine_df=num_of_restaurants_per_cuisine(india_df)
cuisine_df


# In[74]:


sns.barplot(x='cuisine',y='count',data=cuisine_df.sort_values(by='count',ascending=False).head(5))


# In[75]:


cuisine_val = india_df.Cuisines.value_counts()              #values
cuisine_label = india_df.Cuisines.value_counts().index      #labels

plt.pie(x = cuisine_val[:10],labels = cuisine_label[:10],autopct='%1.2f%%',radius=1.5)


# In[76]:


#Top most served cuisines are North India ,Chinese, fast Food ,Mughlai,And Bakery


# In[77]:


len(india_df[india_df['Aggregate_rating']==0])
#india has still 2139 restaurants whose rating is 0 


# In[78]:


len(india_df[india_df['Aggregate_rating']>=4])
#In India 808 restaurants are there whose rating is 4 or above it


# # Now we will explore data for specific city I am going to explore city Mumbai

# In[79]:


mumbai_df=india_df[india_df['City']=='Mumbai']
mumbai_df


# In[80]:


mumbai_df['Cuisines'].value_counts()


# In[81]:


mumbai_cuisine_df=num_of_restaurants_per_cuisine(mumbai_df)
mumbai_cuisine_df


# # Exploring Data for Delhi

# In[82]:


delhi_df=india_df[india_df['City']=='New Delhi']
delhi_df.head()


# In[83]:


delhi_cuisine_df=num_of_restaurants_per_cuisine(delhi_df)
delhi_cuisine_df.sort_values(by='count',ascending=False).head(10)


# In[84]:


#Top 10 cuisines surved by restaurant in New Delhi City are North Indian ,Chinese,Fast Food,Mughlai,Bakery,South Indian,Street Food, Dessert,Italian,Continental


# In[85]:


delhi_df['Has_Online_delivery'].value_counts()


# In[86]:


plt.title('Overall Onilne Delivery Distribution in % in Delhi')
plt.pie(delhi_df['Has_Online_delivery'].value_counts().values,labels=delhi_df['Has_Online_delivery'].value_counts().index,
       autopct='%2f.%%')


# In[87]:


delhi_df['Has_Table_booking'].value_counts()


# In[88]:


sns.scatterplot(x=delhi_df['Aggregate_rating'],y=delhi_df['Average_Cost_for_two'],hue=delhi_df['Has_Online_delivery'])


# In[89]:


sns.scatterplot(x=delhi_df['Aggregate_rating'],y=delhi_df['Average_Cost_for_two'],hue=delhi_df['Has_Table_booking'])


# In[90]:


#Top 10 cuisine offered by restaurant in City 
sns.barplot(x='cuisine',data=delhi_cuisine_df.sort_values(by='count',ascending=False).head(10),y='count')


# In[91]:


#Finding top rated restaurants in Delhi
len(delhi_df[delhi_df['Aggregate_rating']>=3.5])
#overall 1456 restaurants are available in Delhi having ratings >=3.5


# In[92]:


len(delhi_df[delhi_df['Aggregate_rating']==0])
#There are 1425 restaurants are whose rating is 0


# In[93]:


#Finding the restaurants in Delhi which is most visited 
len(delhi_df[delhi_df['Votes']>=5000])
#There are 2 restaurnats in Delhi which is mostly visited by people i.e greater than 5000


# In[94]:


len(delhi_df[(delhi_df['Votes']>=1000)&(delhi_df['Votes']<=4000)])
#113 Restaurants are there in Delhi which is visited by 1000 to 4000 peaple


# In[95]:


len(delhi_df[delhi_df['Votes']==0])
#In Delhi there are still some restaurants which is going to be open or not yet opened and not visited by single person


# In[96]:


len(delhi_df['Is delivering now']=='Yes')
#Currently in Delhi 5473 Restaurants are delivering their serivce 


# In[97]:


india_df.head()


# In[98]:


#For our analysis some columns are not required and testual data must be converted into numerical format ,restaurant name ,add,switch to order menu are not required so we can drop it 
india_df.drop(columns=['Restaurant_Name','Switch to order menu'],inplace=True)


# In[99]:


india_df.drop(columns=['City'],inplace=True)


# In[100]:


india_df.drop(columns=['index'],inplace=True)


# In[101]:


india_df.head()


# In[102]:


#Finding the corelation between the variables
corelation=india_df.corr()
sns.heatmap(corelation,annot=True)


# In[103]:


#Above plot does not show some categorical features in relation so we have to encode that lables into numeric format 
india_df['Has_Online_delivery'].value_counts()


# In[104]:


india_df['Has_Online_delivery'].replace([{'Yes':1},{'No':0}])


# In[105]:


india_df['Has_Online_delivery'].value_counts()


# In[ ]:




