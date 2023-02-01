#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load necessary Libraries
import numpy as np
import pandas as pd


# In[2]:


# Rad all Datasets
books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')


# In[3]:


books.head()


# In[4]:


users.head()


# In[5]:


ratings.head()


# In[6]:


print(books.shape)
print(users.shape)
print(ratings.shape)


# In[7]:


# Ckecking for null values
books.isnull().sum()


# In[8]:


# Ckecking for null values
users.isnull().sum() # Age column has maximum null values but we can ignore it as there is no need in our model


# In[9]:


# Ckecking for null values
ratings.isnull().sum()


# In[10]:


# ckecking duplicate values
books.duplicated().sum()


# In[11]:


# ckecking duplicate values
users.duplicated().sum()


# In[12]:


# ckecking duplicate values
ratings.duplicated().sum()


# # Popularity based recommender system
# 

# we will display the top 50 books with maximum average rating and we will consider only those books which contains minimum 250 votes.
# 

# In[13]:


# Merge datasets named ratings and books and stored in ratings_with_name
ratings_with_name=ratings.merge(books,on='ISBN')
ratings_with_name


# In[14]:


# to get total rating on a particular book
num_rating_df=ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df


# In[15]:


# to get average rating on a particular book

avg_rating_df=ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_ratings'},inplace=True)
avg_rating_df


# In[16]:


# merge two dataframes named num_rating_df and avg_rating_df
popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[17]:


# to get books which has minimum 250 ratings and highest average ratings
popular_df=popular_df[popular_df['num_ratings']>=250].sort_values('avg_ratings',ascending=False).head(50)


# In[18]:


popular_df


# In[19]:


popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_ratings']]
popular_df


# # Collaborative Filtering Based Recommender System
# 

# In this system, we will consider only those users who rate on minimum 200 books and we will consider only those books which have minimum 50 rates. 

# In[23]:


# to get user-IDs who give over 200 ratings 
x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
padhe_likhe_users = x[x].index


# In[24]:


# to get those users who are in padhe_likhe_users and stored in dataframe named filtered_rating
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[25]:


# to get books who have minimum 50 ratings and stored
y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index
famous_books


# In[26]:


# to get those books who are in famous_books and stored in dataframe named final_ratings

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[28]:


# make pivot table where index contain books and columns contain user-IDs
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[29]:


pt.fillna(0,inplace=True)


# In[26]:


pt


# In[27]:


# load libraries for distances

from sklearn.metrics.pairwise import cosine_similarity


# In[28]:


similarity_score = cosine_similarity(pt)


# Here we derive a function which get book name and return 5 most similar books

# In[29]:


def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0] # to get index for given book name
    similar_iteams = sorted(list(enumerate(similarity_score[index])),key = lambda x:x[1],reverse = True)[1:6] # to get 5 most similar books's index and distances for given book name
    for i in similar_iteams: # to get book names of 5 most similar books
        print(pt.index[i[0]])
    


# In[30]:


recommend('The Notebook')


# In[31]:





# In[ ]:




