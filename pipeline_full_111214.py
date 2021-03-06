#from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import msd_sql_functions as msd
import spotify_functions as spotify_functions
import numpy as np
import pandas as pd
import requests
from pyechonest import config
from pyechonest import playlist
config.ECHO_NEST_API_KEY="IHQR7II9KIYTTAPKS"
import pyen
en = pyen.Pyen("IHQR7II9KIYTTAPKS")
import time
import graphlab as gl

class CheckModel:
    '''
    This is used to get top current artists and top recommended artists 
    for a user given an item-item similarity model
    '''
    def __init__(self):
        self.m = msd.MSD_Queries()
        self.model = None
        self.user_col = None
        self.item_col = None
        self.listen_col = None
        self.model_cols = None
        self.user_artist_names = None
        self.user_listens = None
        self.recs = None

    def load_model(self, model, model_cols, user_col, item_col, listen_col=None):
        if type(model) == str:
            self.model = gl.load_model(model)
        else:
            self.model = model
        self.model_cols = model_cols
        self.user_col = user_col
        self.item_col = item_col
        self.listen_col = listen_col

    def fit(self, user_id):
        df = self.get_user_id(user_id)
        top_recs = self.get_user_recs(df, user_id)
        return top_recs, df

    def get_user_data(self, user_id):
        # getting user listen data
        self.user_listens = np.array(self.m.get_user_listendata(user_id))
        user_artist_ids = list(self.user_listens[:,1])
        self.user_artist_names = self.m.get_artist_names(user_artist_ids)

        # arranging in dataframe to match model input
        df = pd.DataFrame(self.user_listens)

        df.columns = [self.user_col, self.item_col, self.listen_col] #format from triplets dataset
        if self.listen_col == None:
            df = df[[self.user_col, self.item_col]]

        df = df[self.model_cols] # changing to model format
        return df

    def get_user_recs(self, df, user_id):
        sf = gl.SFrame(df)
        sf[self.user_col] = 'testuser'
        self.recs = self.model.recommend(users = [user_id], new_observation_data = sf, k = -1, exclude_known = False)
        self.recs[self.user_col] = user_id
        top_recs_names = self.m.get_artist_names(list(self.recs['artist_id'][:20]))
        return top_recs_names


class GroupRecommender:
    '''
    This class is used to take in multiple users data and output artists that all users would enjoy
    '''
    def __init__(self):
        self.m = msd.MSD_Queries()
        self.model = None
        self.user_col = None
        self.item_col = None
        self.listen_col = None
        self.model_cols = None
        self.recs = None
        self.top_items_leastmisery = None
        self.sorted_top_item_scores = None
        self.user_listens = None

    def load_model(self, model, model_cols, user_col, item_col, listen_col=None):
        if type(model) == str:
            self.model = gl.load_model(model)
        else:
            self.model = model
        self.model_cols = model_cols
        self.user_col = user_col
        self.item_col = item_col
        self.listen_col = listen_col


    def fit(self, user_ids):
        tot_recs = self.create_user_rec_sf(user_ids)
        least_misery_list = self.least_misery_list(tot_recs)
        return least_misery_list

    def create_user_rec_sf(self, user_ids):
        '''
        INPUT: MSD user id list
        OUTPUT: recs for those users
        '''

        df = self.get_all_user_listens(user_ids)
        sf = gl.SFrame(df)
        ''' may want to check to ensure that doing all at once is equivalent to getting separate recs'''
        self.recs = self.model.recommend(users = user_ids, new_observation_data = sf, k = -1, exclude_known = False)
        return self.recs

    def create_user_rec_spotify(self, user_ids):
        '''
        INPUT: spotify_user_ids
        OUTPUT: sf of recommendations for that user
        '''
        s = spotify_functions.SpotifyFunctions()
        df_pipeline_list = []
        for user_id in user_ids:
            df_pipeline_user = s.fit(user_id)
            # creating appropriate column structure
            if self.listen_col:
                df_pipeline_user.columns = [self.user_col, self.item_col, self.listen_col]
            else:
                df_pipeline_user = df_pipeline_user[df_pipeline_user.columns[:2]]
                df_pipeline_user.columns = [self.user_col, self.item_col]

            # appending to df_pipeline_list to aggregate
            df_pipeline_list.append(df_pipeline_user)

        df_pipeline_full = pd.concat(df_pipeline_list).reset_index()
        sf = gl.SFrame(df_pipeline_full)
        ''' may want to check to ensure that doing all at once is equivalent to getting separate recs'''
        self.recs = self.model.recommend(users = user_ids, new_observation_data = sf, k = -1, exclude_known = False)
        return self.recs



    def get_all_user_listens(self, user_ids):
        '''
        Input: list of userids
        Output: df of concatenated user listen data
        '''
        # getting user listen data
        user_listens = np.array(self.m.get_user_listendata(user_ids))
        self.user_listens = user_listens

        # arranging in dataframe to match model input
        df = pd.DataFrame(user_listens)

        df.columns = [self.user_col, self.item_col, self.listen_col] #format from triplets dataset
        if self.listen_col == None:
            df = df[[self.user_col, self.item_col]]

        df = df[self.model_cols] # changing to model format
        return df



    def least_misery_list(self, all_recs_sf, top_n_users = 100, top_n_cluster = 100, item_col_lm = 'artist_id', user_col_lm = 'user', score_col = 'rank'):
        '''
        INPUT: sf with ranks
            - top_n_user = top songs to look at for each user
            - top_n_cluster = top songs to cluster
        OUTPUT: ranked song_id sf with least misery score
        '''
        top_items = []
        top_item_scores = []
        users = all_recs_sf[user_col_lm].unique()

        # getting top artists for each user it top_items
        for user in users:
            user_sf = all_recs_sf[all_recs_sf[user_col_lm] == user]
            user_items = user_sf[item_col_lm].unique()
            top_items = top_items + list(user_items[:top_n_users]) 
            # make sure this is getting the top artists
            # maybe sort it?

        # remove duplicates
        top_items = list(set(top_items))

        # go through each artist in top artists, get least misery score
        for item in top_items:
            ''' IMPORTANT! change to min if using score instead of rank '''
            least_misery_score = all_recs_sf[all_recs_sf[item_col_lm] == item][score_col].max()
            top_item_scores.append(least_misery_score)

        # get top artists based on least misery score
        ''' IMPORTANT! change to max if using score instead of rank '''
        idx = np.argsort(top_item_scores)[:top_n_cluster]
        sorted_top_item_scores = np.array(top_item_scores)[idx]
        top_items_leastmisery = np.array(top_items)[idx]
        self.top_items_leastmisery = top_items_leastmisery
        self.sorted_top_item_scores = sorted_top_item_scores
        df = pd.DataFrame(np.hstack((np.array(top_items_leastmisery).reshape(-1,1),
            np.array(sorted_top_item_scores).reshape(-1,1))))
        df.columns = [item_col_lm, score_col]
        return top_items_leastmisery, sorted_top_item_scores, df

class PlaylistRecommender:
    '''
    This class is intended to take in groups of artists, and output playlists
    '''
    def __init__(self):
        self.playlist_list = None
        self.playlist_id_list = None

    def fit_one(self, artist_ids):
        playlist_code = en.get('playlist/basic', artist_id = artist_ids, type = 'artist-radio')
        playlists = []
        playlist_ids = []
        for song in playlist_code['songs']:
            playlists.append('artist: %s, song: %s' % (song['artist_name'], song['title']))
            playlist_ids.append(song['id'])
        return playlists, playlist_ids

    def fit_multiple(self, artist_id_lists):
        '''
        INPUT: list of artist_id lists
        OUTPUT: None
        - creates playlists for each list of artist ids, saves to PlaylistRecommender instance
        '''
        self.playlist_list = []
        self.playlist_id_list = []
        for artist_id_list in artist_id_lists:
            playlist, playlist_ids = self.fit_one(artist_id_list)
            self.playlist_list.append(playlist)
            self.playlist_id_list.append(playlist_ids)

    def fit_transform(self, artist_ids):
        pass

    def print_playlists(self):
        for idx, playlist in enumerate(self.playlist_list):
            print 'Playlist %s:' % str(idx)
            for song in playlist:
                print song


class ArtistTermCluster:
    '''
    INPUT: list of artist_ids
    OUTPUT: cluster labels for each artist
    '''
    def __init__(self):
        self.m = msd.MSD_Queries()
        self.labels = None
        self.artist_ids = None
        self.n_clusters = None

    def fit(self, artist_ids, n_clusters = 8):
        term_docs = self.get_terms(artist_ids)
        labels = self.vectorize(term_docs)
        return labels

    def get_terms(self, artist_ids):
        '''
        INPUT: list of artist_ids
        OUTPUT: dataframe with artist_ids and get_terms
        '''
        self.artist_ids = artist_ids
        if type(artist_ids) != list:
            artist_ids = [artist_ids]
        term_docs = []
        for artist_id in artist_ids:
            terms = self.m.gen_query(select_columns = '*', table = 'artist_term', 
                filter_column = 'artist_id', filter_values = artist_id)
            #print terms
            if len(terms) > 0:
                terms_arr = np.array(terms)[:,1]
                terms_list = [i.replace(' ', '_') for i in terms_arr]
                doc = ' '.join(terms_list)
                term_docs.append(doc)
            else:
                term_docs.append('no_terms')
        return term_docs


    def vectorize(self, term_docs, n_clusters = 8):
        self.n_clusters = n_clusters
        tf = TfidfVectorizer()
        X = tf.fit_transform(term_docs)
        km = KMeans(n_clusters = n_clusters)
        x = km.fit_transform(X)
        self.labels = km.labels_
        return km.labels_

    def create_clusters(self, labels, artist_ids, least_misery_ranked_df, item_col = 'artist_id', score_col = 'rank',
        cluster_col = 'cluster', df_cols = ['artist_id', 'artist_name', 'cluster']):
        '''
        INPUT: 
        - list of artist_ids to cluster
        - recommendation df with scores for each artist
        - 
        '''
        artist_names = np.array(self.m.get_artist_names(artist_ids))
        labels = np.array(labels).reshape(-1,1)
        artist_ids = np.array(artist_ids).reshape(-1,1)
        combo = np.hstack((artist_names, labels))
        df = pd.DataFrame(combo)
        df.columns = df_cols
        df = df.merge(least_misery_ranked_df, how = 'inner', on = item_col)
        df[[cluster_col, score_col]] = df[[cluster_col, score_col]].astype(float)
        df = df.sort([cluster_col, score_col])
        return df

    def get_playlist_seeds(self, df_clusters, cluster_col = 'cluster'):
        '''
        INPUT: df with clusters, artists in each cluster, and the least misery ranking of each artist
        OUTPUT: np array with row = rank of cluster, top 5 artist ids per cluster
        '''
        cluster_dict = {}
        for cluster in df_clusters[cluster_col].unique():
            top_cluster_df = df_clusters[df_clusters[cluster_col] == cluster].head(5)
            cluster_dict[cluster] = {}
            cluster_dict[cluster]['avg_rank'] = np.mean(top_cluster_df['rank'])
            cluster_dict[cluster]['artist_ids'] = np.array(top_cluster_df['artist_id'])
            cluster_dict[cluster]['artist_names'] = np.array(top_cluster_df['artist_name'])
            cluster_dict[cluster]['artist_ranks'] = np.array(top_cluster_df['rank'])
            # might want to add in getting artist terms here

        ''' below: this is super hacky, improve later? '''
        df_cluster_rank = pd.DataFrame(pd.DataFrame(cluster_dict).T['avg_rank'].astype(float)).sort('avg_rank').reset_index().reset_index()
        df_cluster_rank.columns = ['cluster_rank', 'cluster', 'avg_rank']

        cluster_order = df_cluster_rank['cluster'].values
        playlist_seeds = []
        for cluster in cluster_order:
            playlist_seeds.append(list(cluster_dict[cluster]['artist_ids']))
        return playlist_seeds, df_cluster_rank, cluster_dict

class Pipeline:
    '''
    Complete pipeline from usernames to playlist recommendation
    '''
    def __init__(self, model, model_cols, user_col, item_col, listen_col=None):
        self.model = model
        self.model_cols = model_cols
        self.user_col = user_col
        self.item_col = item_col
        self.listen_col = listen_col
        self.df_least_misery = None
        self.df_cluster_rank = None
        self.playlist_recommender = None
        self.playlist_list = None
        self.playlist_id_list = None
        self.playlist_seeds = None

    def fit(self, user_id_list):
        start = time.time()

        least_misery_list = self.get_group_rec(user_id_list)
        lm = time.time()
        print 'Least misery list calculated in: ', lm - start

        self.cluster(least_misery_list)
        cl = time.time()
        print 'Clustering completed in: ', cl - lm

        self.recommend_playlists()
        self.playlist_recommender.print_playlists()
        end = time.time()
        print 'Total time to completion: ', end - start


    def get_group_rec(self, user_id_list):
        '''
        INPUT: list of user spotify ids
        OUTPUT: list of artists that cause least misery to all users
        '''

        gr = GroupRecommender()
        gr.load_model(self.model, model_cols = self.model_cols, user_col = self.user_col, 
            item_col = self.item_col, listen_col = self.listen_col)
        tot_recs = gr.create_user_rec_spotify(user_id_list)
        tot_recs_df = tot_recs.to_dataframe()
        least_misery_list, top_item_scores, df_least_misery = gr.least_misery_list(tot_recs_df)
        self.df_least_misery = df_least_misery
        return least_misery_list

    def cluster(self, least_misery_list):
        '''
        Creates artist clusters from artists that cause least misery, ranks the clusters by top 5 artists,
        and passes those top 5 artists to be playlist seeds
        '''
        #import artist_term_clustering as a #causes graphlab to crash because KMeans is imported
        atc = ArtistTermCluster()
        labels = atc.fit(list(least_misery_list))
        df_clusters = atc.create_clusters(labels, least_misery_list, self.df_least_misery)
        playlist_seeds, df_cluster_rank, cluster_dict = atc.get_playlist_seeds(df_clusters)
        self.playlist_seeds = playlist_seeds
        self.df_cluster_rank = df_cluster_rank

    def recommend_playlists(self):
        '''
        Takes playlist seeds, sends to Echonest API, and returns recommended playlists
        '''
        pr = PlaylistRecommender()
        pr.fit_multiple(self.playlist_seeds)
        self.playlist_list = pr.playlist_list
        self.playlist_id_list = pr.playlist_id_list
        self.playlist_recommender = pr




if __name__ == '__main__':
    start = time.time()
    my_id = '1248440864'
    liza_id = '1299323226'
    userlist = [my_id, liza_id]
    model = gl.load_model('artist_sim_model_triplets')
    pl = Pipeline(model, model_cols = ['user','artist_id','play_count'], 
        user_col = 'user', item_col = 'artist_id', listen_col = 'play_count')
    pl.fit(userlist)


    # user0 = '000c2074db9bed50913055c8cbe847709b6d3235'
    # user1 = '0015abb5e0ed5d63f20dffe7a350d00fee1e9977'
    # user2 = '00127be0d0a5f735e43ea80a9d172116e20ab886'
    # userlist = [user0, user1, user2]
    # start = time.time()
    # my_id = '1248440864'
    # liza_id = '1299323226'
    # userlist = [my_id, liza_id]
    # gr = GroupRecommender()
    # gr.load_model('artist_sim_model_triplets', model_cols = ['user','artist_id','play_count'], 
    #     user_col = 'user', item_col = 'artist_id', listen_col = 'play_count')
    # model_load = time.time()
    # print 'Model Loaded! In: ', model_load - start
    # # tot_recs = gr.create_user_rec_sf(userlist)
    # tot_recs = gr.create_user_rec_spotify(userlist)
    # tot_recs_df = tot_recs.to_dataframe()
    # ind_recs = time.time()
    # print 'Individual Recs Made! In: ', ind_recs - model_load
    # least_misery_list2, top_item_scores, df_least_misery = gr.least_misery_list(tot_recs_df)
    # import artist_term_clustering as a #causes graphlab to crash because KMeans is imported
    # atc2 = a.ArtistTermCluster()
    # labels = atc2.fit(list(least_misery_list2))
    # df_clusters = atc2.create_clusters(labels, least_misery_list2, df_least_misery)
    # playlist_seeds, df_cluster_rank, cluster_dict = atc2.get_playlist_seeds(df_clusters)
    # print df_clusters
    # print playlist_seeds
    # pr = PlaylistRecommender()
    # pr.fit_multiple(playlist_seeds)
    # pr.print_playlists()
    # end = time.time()
    # print 'Total Time to Completion: ', end - start




