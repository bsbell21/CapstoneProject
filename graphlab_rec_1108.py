import graphlab as gl
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import psycopg2

'''
Step 1. Take user ranked list of artists, create least misery ranked list
Step 2. Take top n artists from least misery list, filter user listen data for just those 
    artists, pivot the table to get artist user vectors, and binarize (or normalize 
        if you have time)
    - tranform to dataframe to pivot, then do .values and DBSCAN
    - try with predicted scores as well
Step 3. Run k-means on artist vectors (Or DBSCAN might be better)
Step 4. Rank clusters by top 5 artists in cluster
Step 5. 

'''

def least_misery_list(all_recs_sf, top_n_users = 200, top_n_cluster = 200, item_col = 'song_id', user_col = 'user_id', score_col = 'rank'):
    '''
    INPUT: sf with ranks
        - top_n_user = top songs to look at for each user
        - top_n_cluster = top songs to cluster
    OUTPUT: ranked song_id sf with least misery score
    '''
    top_items = []
    top_item_scores = []
    users = all_recs_sf[user_col].unique()

    # getting top artists for each user it top_items
    for user in users:
        user_sf = all_recs_sf[all_recs_sf[user_col] == user]
        user_items = user_sf[item_col].unique()
        top_items = top_items + list(user_items[:top_n_users]) 
        # make sure this is getting the top artists
        # maybe sort it?

    # remove duplicates
    top_items = list(set(top_items))

    # go through each artist in top artists, get least misery score
    for item in top_items:
        ''' IMPORTANT! change to min if using score instead of rank '''
        least_misery_score = all_recs_sf[all_recs_sf[item_col] == item][score_col].max()
        top_item_scores.append(least_misery_score)

    # get top artists based on least misery score
    ''' IMPORTANT! change to max if using score instead of rank '''
    idx = np.argsort(top_item_scores)[:top_n_cluster]
    top_items_leastmisery = np.array(top_items)[idx]

    return top_items_leastmisery

def get_item_vectors(sf, least_misery_list, binarize = True, item_col = 'song_id', user_col = 'user_id', listen_col = 'listen_count'):
    '''
    INPUT: sf with all user listen data, list of songs with least misery
    OUTPUT: 
    1. array with rows for each item, columns for each user (w/ binary data?)
    2. song_id list corresponding to row in array
    '''
    sf = sf.filter_by(gl.SArray(list(least_misery_list)), column_name = item_col)
    if binarize:
        sf[listen_col] = 1
    df = sf.to_dataframe()
    pivot = df.pivot(item_col, user_col, listen_col).fillna(0)
    return pivot.values, list(pivot.index)



def connect_sql():
    conn = psycopg2.connect(dbname='msd', user='postgres', host='/tmp')
    c = conn.cursor()
    return conn, c

def get_song_info(song_id):
    '''
    INPUT: song id or list of song_ids
    OUTPUT: list of tuples with song name and artist name
    '''
    conn, c = connect_sql()
    c.execute(
    '''
    SELECT artist_name, title FROM songs WHERE song_id IN %s;
    '''
    % (tuple(song_id)))
    conn.commit()
    return c.fetchOne()
    








if __name__ == '__main__':
    pass