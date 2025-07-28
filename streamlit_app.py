#!/usr/bin/env python
# coding: utf-8

# In[461]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[462]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies.head(1)


# In[463]:


credits.head(1)


# In[464]:


movies=movies.merge(credits,on='title')


# In[465]:


movies.head(1)


# In[466]:


movies.info()


# In[467]:


movies=movies[['movie_id','title','genres','keywords','overview','release_date','runtime','cast','crew']] 
movies.head()


# In[468]:


movies.isnull().sum()


# In[469]:


movies.dropna(inplace=True)


# In[470]:


movies.duplicated().sum()


# In[471]:


movies.iloc[0].genres


# In[472]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','Adventure','Fantasy','Science Fiction']


# In[473]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L



# In[474]:


movies['genres']= movies['genres'].apply(convert)


# In[475]:


movies.head()


# In[476]:


movies['keywords']=movies['keywords'].apply(convert)


# In[477]:


movies.head()


# In[478]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L       


# In[479]:


movies['cast']=movies['cast'].apply(convert3)


# In[480]:


movies.head()


# In[481]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']== 'Director':
          L.append(i['name'])
          break
    return L



# In[482]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[483]:


movies.head()


# In[484]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[485]:


movies.head()


# In[486]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[487]:


movies.head()


# In[488]:


movies['release_date']=movies['release_date'].apply(lambda x:x.split())


# In[489]:


movies.head()


# In[490]:


movies=movies[['movie_id','title','genres','keywords','overview','release_date','cast','crew']] 
movies.head()


# In[491]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']+movies['release_date']


# In[492]:


movies.head()


# In[493]:


new_df=movies[['movie_id','title','tags']]
new_df


# In[494]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[495]:


new_df.head()


# In[496]:


import nltk


# In[497]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[506]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[512]:


new_df['tags']=new_df['tags'].apply(stem)


# In[499]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[500]:


new_df.head()


# In[513]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[514]:


vectors=cv.fit_transform(new_df['tags']).toarray()
vectors


# In[515]:


vectors[0]


# In[516]:


cv.get_feature_names_out()


# In[517]:


new_df['tags'][0]


# In[518]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron 2009-12-10')


# In[525]:


from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)
similarity


# In[532]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[558]:


def recommend(movie):
    movie_index=new_df[new_df['title']== movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:7]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[569]:


recommend('Avatar')


# In[550]:


new_df.iloc[0].title

