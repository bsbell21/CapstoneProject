#!/usr/bin/env python
# -*- coding: utf-8 -*-


# coding: utf-8

# In this notebook we will import [GraphLab Create](/products/create) and use it to
# 
# - download data from Amazon S3 containing information about songs that users are listening to
# - train two models that can be used for recommending new songs to users 
# - compare the performance of the two models
# 
# **Note: This notebook uses GraphLab Create 1.0.**

# In[1]:

import graphlab as gl
# set canvas to show sframes and sgraphs in ipython notebook
gl.canvas.set_target('ipynb')


# After importing GraphLab Create, we can download data directly from S3. We have placed a preprocessed version of the [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/) on S3. This data set was used for a [Kaggle challenge](https://www.kaggle.com/c/msdchallenge) and includes data from [The Echo Nest](http://the.echonest.com/), [SecondHandSongs](http://www.secondhandsongs.com/), [musiXmatch](http://musixmatch.com/), and [Last.fm](http://www.last.fm/). This file includes data for a subset of 10000 songs.

# In[2]:

train_file = 'http://s3.amazonaws.com/GraphLab-Datasets/millionsong/10000.txt'

# The below will download a 118 MB file.
sf = gl.SFrame.read_csv(train_file, header=False, delimiter='\t', verbose=False)
sf.rename({'X1':'user_id', 'X2':'song_id', 'X3':'listen_count'}).show()


# In order to evaluate the performance of our model, we randomly split the observations in our data set into two partitions: we will use `train_set` when creating our model and `test_set` for evaluating its performance.

# In[3]:

(train_set, test_set) = sf.random_split(0.8, seed=1)


# One typically wants to initially create a simple recommendation system that can be used as a baseline and to verify that the rest of the pipeline works as expected. The `recommender` package has several models available for this purpose. For example, we can create a model that predicts songs based on their overall popularity across all users.

# In[4]:

popularity_model = gl.popularity_recommender.create(train_set, 'user_id', 'song_id')


# [Collaborative filtering](http://en.wikipedia.org/wiki/Collaborative_filtering) methods make predictions for a given user based on the patterns of other users' activities. One common technique is to compare items based on their [Jaccard](http://en.wikipedia.org/wiki/Jaccard_index) similarity.
# This measurement is a ratio: the number of items they have in common, over the total number of distinct items in both sets.
# We could also have used another slightly more complicated similarity measurement, called [Cosine Similarity](http://en.wikipedia.org/wiki/Cosine_similarity). In the following code block, we compute all the item-item similarities and create an object that can be used for recommendations.

# In[5]:

item_sim_model = gl.item_similarity_recommender.create(train_set, 'user_id', 'song_id')


# It's straightforward to use GraphLab to compare models on a small subset of users in the `test_set`. The [precision-recall](http://en.wikipedia.org/wiki/Precision_and_recall) plot that is computed shows the benefits of using the similarity-based model instead of the baseline `popularity_model`: better curves tend toward the upper-right hand corner of the plot. 
# 
# The following command finds the top-ranked items for all users in the first 500 rows of `test_set`. The observations in `train_set` are not included in the predicted items.

# In[6]:

result = gl.recommender.util.compare_models(test_set, [popularity_model, item_sim_model],
                                            user_sample=.1, skip_set=train_set)


# Now let's ask the item similarity model for song recommendations on several users. We first create a list of users and create a subset of observations, `users_ratings`, that pertain to these users.

# In[7]:

K = 10
users = gl.SArray(sf['user_id'].unique().head(100))


# Next we use the `recommend()` function to query the model we created for recommendations. The returned object has four columns: `user_id`, `song_id`, the `score` that the algorithm gave this user for this song, and the song's rank (an integer from 0 to K-1). To see this we can grab the top few rows of `recs`:

# In[8]:

recs = item_sim_model.recommend(users=users, k=K)
recs.head()


# To learn what songs these ids pertain to, we can merge in metadata about each song.

# In[9]:

# Get the meta data of the songs

# The below will download a 75 MB file.
songs = gl.SFrame.read_csv('http://s3.amazonaws.com/GraphLab-Datasets/millionsong/song_data.csv', verbose=False)
songs = songs[['song_id', 'title', 'artist_name']]
results = recs.join(songs, on='song_id', how='inner')

# Populate observed user-song data with song info
userset = frozenset(users)
ix = sf['user_id'].apply(lambda x: x in userset, int)  
user_data = sf[ix]
user_data = user_data.join(songs, on='song_id')[['user_id', 'title', 'artist_name']]


# In[10]:

# Print out some recommendations 
for i in range(5):
    user = list(users)[i]
    print "User: " + str(i + 1)
    user_obs = user_data[user_data['user_id'] == user].head(K)
    del user_obs['user_id']
    user_recs = results[results['user_id'] == str(user)][['title', 'artist_name']]

    print "We were told that the user liked these songs: "
    print user_obs.head(K)

    print "We recommend these other songs:"
    print user_recs.head(K)

    print ""


# (Looking for more details about the modules and functions? Check out the <a href="/products/create/docs/">API docs</a>.)
