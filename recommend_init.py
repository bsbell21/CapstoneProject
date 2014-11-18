import graphlab as gl
import numpy as np

'''

Notes

1. Take PCA and then KMeans on the User Listen data
2. Try artist-artist similarity on all the data
3. artist --> term matrix clustering
- make each artist a document, do TFIDF on terms

4. create network from all artist data
5. query echonest for term with weight
6. graph based
- could connect artists as vertices with edges as terms

TODO:
- Get artist from MSD for each song
- Figure out how to access from the cloud

to CHECK OUT:
- there's a 
New plan:
1. Get MSD listening data into SQL or database of some kind
2. Get full data for MSD listening data for songs
3. sum up to genre

Potential iteration 1:
- get genre similarity directly from echonest
- create genre similarity matrix
- for each user in 120k dataset, sum up to genre their preferences
- multiply through similarity matrix, get new genre prefs??
- or maybe even skip genre/genre similarity matrix, just do top genres for each
- need to find a way to group the genres
- could just take top genre for one, look for genre in second that
    is above X threshold of similarity
- create second genre that 

Potential iteration 1 idea 2, genre:
- make genre genre similarity matrix
- for each user, sum up to genre their preferences by:
    - multiplying playcount/rating by the genre vector of the artist
    - if artist has multiple genres, average the genre vector to get
    artist genre vector
    - sum up to get genre preference vector for user??
    - then what?? rank the genres and cluster
    - i could cluster the songs based on genre vecotr, assign each song a cluster name
    - take the centroids of each and rank the centroids for each user
    - take centroids that would cause the least misery, and are most highly correlated
        - how does this work exactly???

    - then go back to the songs that are in those centroids
    - rank them for each user
    - pick the top 2
    - put them into the playlist generator

or
- just use artist info, sum up to that level
- do artist artist similarity
    - could potentially use echo nest for this as well
- get top artists for each user, with scores, maybe don't even do artist-artist
    similarity yet, just straight top artists
- compare top artists across 
- group aggregation
    1)
    - find artist that will cause least misery
    - find top artists that are most similar with some threshold from other
        users
    - group together for playlist 1
    - calculate total happiness for x artists
    - find next artist that will cause least misery, that is also unsimilar
        with some threshold from the first group
    - rinse, repeat

    2) 

PROBLEMS
- can get genre by artist, but not sure if I can by song
- can get genre from Gracenote
- genre both by artist and song come in multiples... 
    - could binarize genres out..?
    - could just pick top genre... probably what I should do to start
- as a first pass just connect artist to top genre?




Dealing with big data:
- graphlab might be slow
- could try to do it in sql - madlib
- could try to use EMR in Amazon




To Do:
1. Normalize every rating scale from 1 to 5, 20% of each/by ranking
2. Create item-item similarity matrix
3. Do collaborative filter
4. sum up to artist level
5. get data for songs
6. get spotify, soundcloud data

- get featured playlist data from spotify
- get data from spotify
- check out igraph
- get playlist data from echonest, query the shit out of it? look at
    co-occurence


Questions:
1. How do you get a recommendation for a new user with matrix factorization?
2. 
'''

'''
Upgraded echnoest data:

Your API Key: IHQR7II9KIYTTAPKS 

Your Consumer Key: 368f158957a9b30e8844bddca8d5b222 

Your Shared Secret: xFSrM/KBTeisHQ9CHHdwew

'''

from pyechonest import config
config.ECHO_NEST_API_KEY="IHQR7II9KIYTTAPKS"
from pyechonest import artist
from pyechonest import playlist
from pyechonest import catalog
import pyen
en = pyen.Pyen("IHQR7II9KIYTTAPKS")
from pymongo import MongoClient
import pandas as pd
import time

def create_genre_similarity_matrix(start_genre_num=0):
    '''
    will pull from echonest every genre and the similarity scores
    for the closest genres to it
    '''

    genre_sim_dic = {}
    for genre in en.get('genre/list', results = 2000)['genres'][start_genre_num:]:
        name = genre['name']
        genre_sim_dic[name] = {}
        genre_sim = en.get('genre/similar', name = genre['name'])['genres']
        for g in genre_sim:
            genre_sim_dic[name][g['name']] = g['similarity'] 
        genre_sim_dic[name][name] = 1
        print genre
    return genre_sim_dic

def get_cat_artistgenre(cat_id):
    cat = catalog.Catalog(cat_id)
    cat_items = cat.get_item_dicts()
    for idx, song in enumerate(cat_items):
        artist_id = song['artist_id']
        genre_list = en.get('artist/profile', 
            id = artist_id, bucket = 'genre')['artist']['genres']

        cat_items[idx]['artist_genres'] = genre_list


def import_cat_data():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["onefiftyk"]
    collection = db.userlistendata
    df = pd.read_csv('taste_profile_usercat_120k_catalog.txt', header=None)
    print 'Dataframe loaded'
    # cat = catalog.Catalog(df.values[1][0])
    # collection.insert(cat.get_item_dicts())

    i = 0
    while i < len(df.values):
        try:
            cat = catalog.Catalog(df.values[i][0])
            collection.insert(cat.get_item_dicts())
        except:
            i -= 1
            time.sleep(10)
        i += 1

    # for idx, cat_id in enumerate(df.values):
    #     cat = catalog.Catalog(cat_id[0])
    #     collection.insert(cat.get_item_dicts())
    #     if idx % 115 == 0:
    #         time.sleep(60)


        # try:
        #     cat = catalog.Catalog(cat_id[0])
        #     collection.insert(cat.get_item_dicts())
        #     if idx % 115 == 0:
        #         time.sleep(60)
        # except:
        #     print 'error', idx
        #     break
    print '... Success!'
    return idx





def run():
    pass


def load_data(filename):
    pass

def current_top_songs(userid, sf, value_col = 'Count'):
    ''' return a users top listened to songs '''
    return sf[sf['user'] == userid].sort(value_col, ascending = False)

def factor_top_n(song_df, n=10, factor_col = 'factors', item_col = 'artist_song'):
    ''' 
    Returns top songs for each latent factor from matrix factorization
    song_df: dataframe of songs w/ each factor score w/ label 'factors'
    '''
    factor_topn = []
    factor_songs = np.vstack(np.array(song_df[factor_col])).T
    for factor in factor_songs:
        top_song_idx = np.argsort(factor)[:-n:-1]
        top_songs = song_df[item_col][top_song_idx].values
        factor_topn.append(top_songs)

    return factor_topn



def normalize_by_column(sf, col_name, value='Count', col_to_norm = 'norm'):
    '''
    Function to normalize a given column
    - value: what column you are normalizing by, normed col divided by this
    - col_to_norm: column that will be normed (ie in the numerator)
    - col_name: col to normalize by, if 'song' col_to_norm would be divided by
        total value count for all 'songs' that were the same
    '''
    total_counts = sf.groupby(col_name,
                        {'total_count':gl.aggregate.SUM(value)})

    avg_count = total_counts['total_count'].mean()
    print avg_count

    grouped_norm = sf.join(total_counts, on = col_name)
    grouped_norm['norm_' + col_name] = grouped_norm[col_to_norm] / grouped_norm['total_cnt']

    del grouped_norm['total_count']
    return grouped_norm

def create_grouped_table(filename, min_listens=25):

    ''' 
    Aggregating last.FM data into user listens/song
    Normalizing table to account for differences in total listens
    Notes:
        - is there a better way to normalize? Is % listens really the best p
        proxy for a user rating?
        - wouldn't there be some way to TFIDF??
        - seems that there should be some confidence measure for people you
        have more listens for, someone who only listened to one song will have 
        100% for that song which seems somewhat biased/will overly boost that song
        - normalizing fully may be overweighing users that don't do much

    Next steps:
        - look into TFIDF: http://graphlab.com/products/create/docs/generated/graphlab.text_analytics.tf_idf.html
        - talk to Jon about alternatives

    '''

    # 1. Explore the data files. To start load them into a GraphLab SFrame.
    song_sf = gl.SFrame.read_csv(filename, delimiter=',', header=None,
                                 error_bad_lines=False)
    song_sf.rename({'X1':'user', 'X2':'timestamp', 'X3': 'aid', 'X4': 'artist',
                    'X5':'sid','X6':'song'})

    # create grouped table
    grouped_s = song_sf.groupby(key_columns=['user', 'artist', 'song'],
                                operations=gl.aggregate.COUNT())

    grouped_s['artist_song'] = grouped_s['artist'] + ': ' + grouped_s['song'] 

    # create normalized table
    total_counts = grouped_s.groupby('user',
                        {'total_cnt':gl.aggregate.SUM('Count')})

    grouped_norm = grouped_s.join(total_counts, on = 'user')
    grouped_norm['norm'] = grouped_norm['Count'] / grouped_norm['total_cnt']

    smaller_s = grouped_s[grouped_s['Count'] > min_listens]

    smaller_norm = grouped_norm[grouped_norm['Count'] > min_listens]

    # create train and test sets
    train_set_s, test_set_s= \
        gl.recommender.util.random_split_by_user(smaller_s,
                                                 user_id='user',
                                                 item_id='song')

    train_set_norm, test_set_norm= \
        gl.recommender.util.random_split_by_user(smaller_norm,
                                                 user_id='user',
                                                 item_id='song')

    return grouped_norm, smaller_norm, train_set_norm, test_set_norm, grouped_s, smaller_s, train_set_s, test_set_s


class SongRecommender_gl:
    def __init__(self):
        self.model = None

    def fit(train_set, user_id = 'user', item_id = 'artist_song', target = 'norm',  binarize = False):
        if binarize:
            train_set[target] = train_set[target] > 0

        self.model = gl.recommender.create(train_set, 
            user_id, item_id, target)

    def get_user_rec_vector(user_id):
        pass
        

