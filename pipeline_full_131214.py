#from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import msd_sql_functions as msd
#import spotify_functions as spotify_functions
import spotify_functions_mult131114 as spotify_functions
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
import ipdb

'''
This differs from pipeline_full_111214 in that it is removing SpotifyFunctions from the pipeline and placing it outside
so tokens need not be called every time
'''

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
        self.all_recs_sf = None
        self.score_col = None

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

    def create_user_rec_from_df(self, df_pipeline):
        '''
        INPUT: DataFrame with User, Artist, Play Count columns
        OUTPUT: SFrame with recommendations for all users
        '''
        # ensuring df_pipeline columns in correct format
        if self.listen_col:
            df_pipeline.columns = [self.user_col, self.item_col, self.listen_col]
        else:
            df_pipeline = df_pipeline[df_pipeline.columns[:2]]
            df_pipeline.columns = [self.user_col, self.item_col]

        # creating recommendations
        user_ids = list(df_pipeline[self.user_col].unique())
        sf = gl.SFrame(df_pipeline)
        ''' may want to check to ensure that doing all at once is equivalent to getting separate recs'''
        self.recs = self.model.recommend(users = user_ids, new_observation_data = sf, k = -1, exclude_known = False)
        return self.recs

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
        INPUT: DataFrame with User, Artist, Play Count columns
        OUTPUT: SFrame with recommendations for all users
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
        self.score_col = score_col
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


        ''' changing score to rank proxy '''
        tot_recs_df = all_recs_sf # converting to dataframe already in pipeline

        tot_recs_df_lm = tot_recs_df[tot_recs_df[item_col_lm].isin(top_items)]
        self.tot_recs_df_lm = tot_recs_df_lm #remove

        # creating "score" proxy from rank by dividing all ranks by max rank in least misery and subtracting from 1
        # later should use score and just normalize
        ''' moving later so max score is of top items only '''
        # max_rank = tot_recs_df_lm[score_col].max()
        # self.max_rank = max_rank #remove
        # tot_recs_df_lm[score_col] = tot_recs_df_lm[score_col].apply(lambda x: 1 - ((1.0*x)/max_rank))

        # getting scores for each user
        tot_recs_lm_pivot = tot_recs_df_lm.pivot(item_col_lm, user_col_lm, score_col)

        # getting least misery score for each artist - would be min if not using rank
        tot_recs_lm_pivot['lm_rank'] = np.max(tot_recs_lm_pivot, axis = 1)

        
        #tot_recs_lm_pivot['rank_score'] = np.min(tot_recs_lm_pivot, axis = 1)

        # sorting and taking top specified artists
        tot_recs_lm_pivot = tot_recs_lm_pivot.sort('lm_rank')[:top_n_cluster]

        # creating rank "score" proxy from rank by dividing all ranks by max rank in least misery and subtracting from 1
        max_rank = tot_recs_lm_pivot['lm_rank'].max()
        tot_recs_lm_pivot = tot_recs_lm_pivot.apply(lambda x: 1 - ((1.0*x)/max_rank)).reset_index()
        tot_recs_lm_pivot = tot_recs_lm_pivot.rename(columns = {'lm_rank':'rank_score'})

        self.tot_recs_lm_pivot = tot_recs_lm_pivot
        df_least_misery = tot_recs_lm_pivot[[item_col_lm, 'rank_score']]
        top_items_leastmisery = df_least_misery[item_col_lm].values
        sorted_top_item_scores = df_least_misery['rank_score'].values
        return top_items_leastmisery, sorted_top_item_scores, df_least_misery, tot_recs_lm_pivot

        





        ''' end of new code'''
        ''' below is code for rank scoring '''
        # users_least_misery_list = [] # making list for each user for whole least misery list
        # # filling users_least_misery_list

        # # for users in users:
        # #     user_least_misery_sf = all_recs_sf[all_recs_sf[user_col_lm]]

        # # go through each artist in top artists, get least misery score
        # for item in top_items:
        #     ''' IMPORTANT! change to min if using score instead of rank '''
        #     least_misery_score = all_recs_sf[all_recs_sf[item_col_lm] == item][score_col].max()
        #     top_item_scores.append(least_misery_score)


        # # get top artists based on least misery score
        # ''' IMPORTANT! change to max if using score instead of rank '''
        # idx = np.argsort(top_item_scores)[:top_n_cluster]
        # sorted_top_item_scores = np.array(top_item_scores)[idx]
        # top_items_leastmisery = np.array(top_items)[idx]
        # self.top_items_leastmisery = top_items_leastmisery
        # self.sorted_top_item_scores = sorted_top_item_scores
        # df = pd.DataFrame(np.hstack((np.array(top_items_leastmisery).reshape(-1,1),
        #     np.array(sorted_top_item_scores).reshape(-1,1))))
        # df.columns = [item_col_lm, score_col]
        # return top_items_leastmisery, sorted_top_item_scores, df


# class ArtistTermCluster:
#     '''
#     INPUT: list of artist_ids
#     OUTPUT: cluster labels for each artist
#     '''
#     def __init__(self):
#         self.m = msd.MSD_Queries()
#         self.labels = None
#         self.artist_ids = None
#         self.n_clusters = None

#     def fit(self, artist_ids, n_clusters = 8):
#         term_docs = self.get_terms(artist_ids)
#         labels = self.vectorize(term_docs)
#         return labels

#     def get_terms(self, artist_ids):
#         '''
#         INPUT: list of artist_ids
#         OUTPUT: dataframe with artist_ids and get_terms
#         '''
#         self.artist_ids = artist_ids
#         if type(artist_ids) != list:
#             artist_ids = [artist_ids]
#         term_docs = []
#         for artist_id in artist_ids:
#             terms = self.m.gen_query(select_columns = '*', table = 'artist_term', 
#                 filter_column = 'artist_id', filter_values = artist_id)
#             #print terms
#             if len(terms) > 0:
#                 terms_arr = np.array(terms)[:,1]
#                 terms_list = [i.replace(' ', '_') for i in terms_arr]
#                 doc = ' '.join(terms_list)
#                 term_docs.append(doc)
#             else:
#                 term_docs.append('no_terms')
#         return term_docs


#     def vectorize(self, term_docs, n_clusters = 8):
#         self.n_clusters = n_clusters
#         tf = TfidfVectorizer()
#         X = tf.fit_transform(term_docs)
#         km = KMeans(n_clusters = n_clusters)
#         x = km.fit_transform(X)
#         self.labels = km.labels_
#         return km.labels_

#     def create_clusters(self, labels, artist_ids, least_misery_ranked_df, item_col = 'artist_id', score_col = 'rank',
#         cluster_col = 'cluster', df_cols = ['artist_id', 'artist_name', 'cluster']):
#         '''
#         INPUT: 
#         - list of artist_ids to cluster
#         - recommendation df with scores for each artist
#         - 
#         '''
#         artist_names = np.array(self.m.get_artist_names(artist_ids))
#         labels = np.array(labels).reshape(-1,1)
#         artist_ids = np.array(artist_ids).reshape(-1,1)
#         combo = np.hstack((artist_names, labels))
#         df = pd.DataFrame(combo)
#         df.columns = df_cols
#         df = df.merge(least_misery_ranked_df, how = 'inner', on = item_col)
#         df[[cluster_col, score_col]] = df[[cluster_col, score_col]].astype(float)
#         df = df.sort([cluster_col, score_col])
#         return df

#     def get_playlist_seeds(self, df_clusters, cluster_col = 'cluster'):
#         '''
#         INPUT: df with clusters, artists in each cluster, and the least misery ranking of each artist
#         OUTPUT: np array with row = rank of cluster, top 5 artist ids per cluster
#         '''
#         cluster_dict = {}
#         for cluster in df_clusters[cluster_col].unique():
#             top_cluster_df = df_clusters[df_clusters[cluster_col] == cluster].head(5)
#             cluster_dict[cluster] = {}
#             cluster_dict[cluster]['avg_rank'] = np.mean(top_cluster_df['rank'])
#             cluster_dict[cluster]['artist_ids'] = np.array(top_cluster_df['artist_id'])
#             cluster_dict[cluster]['artist_names'] = np.array(top_cluster_df['artist_name'])
#             cluster_dict[cluster]['artist_ranks'] = np.array(top_cluster_df['rank'])
#             # might want to add in getting artist terms here

#         ''' below: this is super hacky, improve later? '''
#         df_cluster_rank = pd.DataFrame(pd.DataFrame(cluster_dict).T['avg_rank'].astype(float)).sort('avg_rank').reset_index().reset_index()
#         df_cluster_rank.columns = ['cluster_rank', 'cluster', 'avg_rank']

#         cluster_order = df_cluster_rank['cluster'].values
#         playlist_seeds = []
#         for cluster in cluster_order:
#             playlist_seeds.append(list(cluster_dict[cluster]['artist_ids']))
#         return playlist_seeds, df_cluster_rank, cluster_dict

class ArtistClusterAF:
    def __init__(self):
        self.m = msd.MSD_Queries()
        self.labels = None
        self.artist_ids = None

        self.af = None

    def fit(self, df_lm_allusers, item_col = 'artist_id', score_col = 'rank_score', name_col = 'artist_name'):
        '''
        INPUT: DataFrame with names, ids, and least misery ranking for each artist
        OUTPUT: playlist seeds with rankings
        '''

        df_least_misery = df_lm_allusers.copy()
        df_least_misery = df_least_misery[[item_col, 'rank_score']]


        self.df_cols = df_least_misery.columns
        self.item_col = item_col
        self.score_col = score_col
        self.name_col = name_col
        self.artist_ids = list(df_least_misery[item_col].values)

        # addding names to df least misry
        self.artist_names = np.array(self.m.get_artist_names(self.artist_ids))#[:,1]
        self.artist_name_id_df = pd.DataFrame(self.artist_names)
        self.artist_name_id_df.columns = [item_col, name_col]
        self.df_least_misery = df_least_misery.merge(self.artist_name_id_df, on = item_col)


        self.df_lm_allusers = df_lm_allusers.merge(self.artist_name_id_df, on = item_col)


        term_docs, artists_with_terms_df = self.get_terms(self.artist_ids)

        # removing any artists that don't have terms
        self.df_lm_allusers = self.df_lm_allusers.merge(artists_with_terms_df, on = 'artist_id')
        self.df_least_misery = self.df_least_misery.merge(artists_with_terms_df, on = 'artist_id')

        feat_mtx = self.vectorize(term_docs)
        df_least_misery_clustered = self.cluster(feat_mtx, self.df_lm_allusers)
        playlist_seed_df, playlist_seeds, playlist_seed_names, playlist_seed_scores = self.get_playlist_seeds(df_least_misery_clustered)


        self.df_least_misery_clustered = df_least_misery_clustered # remove

        return playlist_seed_df, playlist_seeds, playlist_seed_names, playlist_seed_scores

        pass

    def get_terms(self, artist_ids):
        '''
        INPUT: list of artist_ids
        OUTPUT: dataframe with artist_ids and get_terms
        '''
        self.artist_ids = artist_ids
        if type(artist_ids) != list:
            artist_ids = [artist_ids]
        term_docs = []

        ''' sql queries don't return in the same order they were sent !!!!'''
        # slower more accurate version
        terms_all = self.m.gen_query(select_columns = 'artist_id, term', table = 'cluster_artist_nontriplets', 
            filter_column = 'artist_id', filter_values = artist_ids)
        start = time.time()
        # terms_all = self.m.gen_query(select_columns = 'artist_id, term', table = 'all_artist_terms', 
        #     filter_column = 'artist_id', filter_values = artist_ids)
        end = time.time()
        print 'terms queried in: ', end - start


        df_terms = pd.DataFrame(terms_all)
        df_terms.columns = ['artist_id', 'term']
        self.df_terms = df_terms
        #getting unique artists left, kinda hacky - change later
        artists_with_terms_df = df_terms[['artist_id']].groupby('artist_id').sum().reset_index()
        self.artists_with_terms_df = artists_with_terms_df


        # just making sure in the same order as df_least_misery
        df_ordering = self.df_least_misery.merge(df_terms, on = 'artist_id')
        df_terms = df_ordering[['artist_id', 'term']]
        
        # need to set sort to False or it will reorder
        term_docs = df_terms.groupby('artist_id', sort = False)['term'].agg(' '.join).values
        self.term_docs_artists = np.array(df_terms.index)
        final = time.time()
        print 'artist term vectors created in: ', final - end

        # may have to get unique values here if not all artist ids guaranteed in databse
        # term_docs = df_terms.groupby('artist_id')['term'].agg(' '.join).values
        
        return term_docs, artists_with_terms_df

    def vectorize(self, term_docs):
        '''
        clustering artists using affinity propogation, returns cluster labels
        '''
        tf = TfidfVectorizer()
        feat_mtx = tf.fit_transform(term_docs)
        return feat_mtx

    def cluster(self, feat_mtx, df_lm_allusers):
        # clustering artists based on AffinityPropogation
        start = time.time()
        af = AffinityPropagation()
        af.fit(feat_mtx)
        self.labels = af.labels_
        self.af = af

        # adding cluster labels to least misery dataframe and sorting by rank and cluster
        #df_least_misery_clustered = self.df_least_misery.copy() --> changing to df_lm_allusers
        print 'number of labels: ', len(self.labels)
        print 'labels', self.labels
        
        # print 'least misery clustered length', len(df_least_misery_clustered)
        
        df_least_misery_clustered = df_lm_allusers.copy()
        print 'len df least misery: ', len(df_least_misery_clustered)
        
        df_least_misery_clustered['cluster'] = self.labels
        df_least_misery_clustered[['cluster', self.score_col]] = df_least_misery_clustered[['cluster', self.score_col]].astype(float)
        ''' will do different sorting if not using rank '''
        # now set to false as looking for highest score
        df_least_misery_clustered = df_least_misery_clustered.sort(['cluster', self.score_col], ascending = False)
        self.df_least_misery_clustered = df_least_misery_clustered
        end = time.time()
        print 'clustering completed in: ', end - start  
        return df_least_misery_clustered

    def get_playlist_seeds(self, df_least_misery_clustered, penalize_less_than = 4, penalization = .3):


        playlist_seed_df_list = []
        avg_rank = []
        for i in np.unique(self.labels):
            top_cluster = df_least_misery_clustered[df_least_misery_clustered['cluster'] == i].head(5)
            playlist_seed_df_list.append(top_cluster)
            # rankmean = top_cluster[self.score_col].mean()
            # # trying to remove clusters with < 4 artists from top
            # if len(top_cluster) < penalize_less_than:
            #     rankmean -= penalization
            # top_cluster['avg_rank_score'] = rankmean
            # avg_rank.append(rankmean)
            # playlist_seed_df_list.append(top_cluster)


        ''' the goal of the next chunk of code is to get ranking for each cluster, penalizing for the number of artists in the cluster '''
        playlist_seed_df = pd.concat(list(playlist_seed_df_list))
        # creating cluster grouping with ranking by the cluster
        psd_group = playlist_seed_df.copy()
        psd_group = psd_group[psd_group.columns - [self.name_col, self.item_col, self.score_col]]
        # setting cluster to index so np.mean will work on all columns but cluster
        psd_group = psd_group.set_index('cluster')
        psd_group['cluster_score'] = np.min(psd_group, axis = 1) # least misery ranking
        #psd_group['cluster_score'] = np.mean(psd_group, axis = 1) # average happiness ranking
        # resetting index and then grouping 
        psd_group = psd_group.reset_index().groupby('cluster')
        # creating table with count data for each cluster so I can penalize small clusters
        self.psd1 = psd_group
        psd_group_count = psd_group.count()[['cluster_score']]
        print 'psd_group count: ', psd_group_count
        # ipdb.set_trace()
        psd_group_count.columns = ['count']
        # getting mean of all other columns and adding count
        print 'psd_group count: ', psd_group_count
        self.psd2 = psd_group
        print 'psd_group before: ', psd_group
        psd_group = pd.merge(psd_group.mean(), psd_group_count, left_index = True, right_index = True)
        print 'psd_group after: ', psd_group
        
        # creating boolean mask for penalization and penalizing
        psd_group['count'] = psd_group['count'].apply(lambda x: x < penalize_less_than)
        print 'psd_group2', psd_group
        psd_group['cluster_score'] = psd_group['cluster_score'] - penalization*psd_group['count']
        # removing count column
        print 'psd_group3', psd_group
        self.psd3 = psd_group
        psd_group = psd_group.drop('count', axis = 1)
        print 'psd_group4', psd_group
        self.psd4 = psd_group
        psd_group = psd_group.sort('cluster_score', ascending = False).reset_index()
        print playlist_seed_df
        print 'psd_group5', psd_group
        self.psd5 = psd_group
        self.playlist_seed_df = playlist_seed_df
        self.psd_group = psd_group

        # adding cluster score to playlist_seed_df
        playlist_seed_df = playlist_seed_df.merge(psd_group[['cluster', 'cluster_score']], on = 'cluster')
        playlist_seed_df = playlist_seed_df.sort('cluster_score', ascending = False)

        ''' end chunk '''

        playlist_seeds = []
        playlist_seed_names = []
        for i in list(psd_group['cluster'].values):
            artist_ids = list(playlist_seed_df[playlist_seed_df['cluster'] == i][self.item_col].values)
            artist_names = list(playlist_seed_df[playlist_seed_df['cluster'] == i][self.name_col].values)
            playlist_seeds.append(artist_ids)
            playlist_seed_names.append(', '.join(artist_names))




        # # put together playlist seed df
        # idx = np.argsort(np.array(avg_rank))[::-1] # reversing to have largest scores first
        # playlist_seed_df_list = np.array(playlist_seed_df_list)[idx]

        # playlist_seeds = [i[self.item_col].values for i in playlist_seed_df_list]
        # playlist_seed_names = [', '.join(list(i[self.name_col].values)) for i in playlist_seed_df_list]
        # playlist_seed_df = pd.concat(list(playlist_seed_df_list))

        return playlist_seed_df, playlist_seeds, playlist_seed_names, psd_group


    def get_playlist_seeds_old(self, df_least_misery_clustered, penalize_less_than = 4, penalization = .3):

        # taking top 5 artists per cluster and making playlist seeds
        playlist_seed_df_list = []
        avg_rank = []
        for i in np.unique(self.labels):
            top_cluster = df_least_misery_clustered[df_least_misery_clustered['cluster'] == i].head(5)
            rankmean = top_cluster[self.score_col].mean()
            # trying to remove clusters with < 4 artists from top
            if len(top_cluster) < penalize_less_than:
                rankmean -= penalization
            top_cluster['avg_rank_score'] = rankmean
            avg_rank.append(rankmean)
            playlist_seed_df_list.append(top_cluster)

        # put together playlist seed df
        idx = np.argsort(np.array(avg_rank))[::-1] # reversing to have largest scores first
        playlist_seed_df_list = np.array(playlist_seed_df_list)[idx]

        playlist_seeds = [i[self.item_col].values for i in playlist_seed_df_list]
        playlist_seed_names = [', '.join(list(i[self.name_col].values)) for i in playlist_seed_df_list]
        playlist_seed_df = pd.concat(list(playlist_seed_df_list))

        return playlist_seed_df, playlist_seeds, playlist_seed_names



class PlaylistRecommender:
    '''
    This class is intended to take in groups of artists, and output playlists
    '''
    def __init__(self):
        self.playlist_list = None
        self.playlist_id_list = None
        self.m = msd.MSD_Queries()

    def fit_one(self, artist_ids):
       # playlist_code = en.get('playlist/basic', artist_id = artist_ids, type = 'artist-radio', bucket = )
        playlist_code = en.get('playlist/static', artist_id = artist_ids , type = 'artist-radio', bucket = ['id:spotify', 'tracks'], limit='true')
        playlists = []
        playlist_ids = []
        playlist_spotify_track_ids = []
        for song in playlist_code['songs']:
            playlists.append('artist: %s, song: %s' % (song['artist_name'], song['title']))
            playlist_ids.append(song['id'])
            spotify_track_id = str(song['tracks'][0]['foreign_id'].split(':')[-1])
            playlist_spotify_track_ids.append(spotify_track_id)
        return playlists, playlist_ids, playlist_spotify_track_ids

    def fit_multiple(self, artist_id_lists):
        '''
        INPUT: list of artist_id lists
        OUTPUT: None
        - creates playlists for each list of artist ids, saves to PlaylistRecommender instance
        '''
        self.playlist_list = []
        self.playlist_id_list = []
        self.playlist_spotify_id_list = []
        self.playlist_names = []
        for artist_id_list in artist_id_lists:
            playlist, playlist_ids, playlist_spotify_track_ids = self.fit_one(artist_id_list)
            self.playlist_list.append(playlist)
            self.playlist_id_list.append(playlist_ids)
            self.playlist_spotify_id_list.append(playlist_spotify_track_ids)
            self.playlist_names.append(self.get_playlist_terms(artist_id_list))

    def fit_transform(self, artist_ids):
        pass

    def print_playlists(self):
        for idx, playlist in enumerate(self.playlist_list):
            print 'Playlist %s:' % str(idx)
            for song in playlist:
                print song

    def print_playlists_withseeds(self, playlist_seed_names, playlist_names, playlist_seed_scores):
        for idx, playlist in enumerate(self.playlist_list):
            print 'Playlist %s, seeded with: %s' % (playlist_names[idx], playlist_seed_names[idx])
            print 'User Scores:'
            for col in playlist_seed_scores.columns:
                print col, ': ', playlist_seed_scores[col].values[idx]

            for song in playlist:
                print song

    def get_playlist_terms(self, artist_ids):
        '''
        INPUT: list of artist_ids seeding playlist
        OUTPUT: Top 3 most common terms for group of artists
        '''
        try:
            terms_all = self.m.gen_query(select_columns = '*', table = 'all_artist_terms', 
                filter_column = 'artist_id', filter_values = artist_ids)


            df_terms = pd.DataFrame(terms_all)
            df_terms.columns = ['artist_id','term']
            df_terms = df_terms[['term']]
            df_terms['count'] = 1
            df_terms_grouped = df_terms.groupby('term').sum().sort('count', ascending = False).reset_index()
            result = '/'.join(list(df_terms_grouped['term'][:3].values))
        except:
            print terms_all
            result = 'wetunes'
        
        return result

    def get_mult_playlist_terms(self, artist_ids_lists):
        pass





class Pipeline:
    '''
    Complete pipeline from df of user listens to playlist recommendation
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
        self.playlist_seed_df = None
        self.playlist_seed_names = None
        self.df_pipeline = None
        self.user_ids = None
        self.least_misery_list = None #should remove
        self.tot_recs_df = None
        self.score_col = None
        self.tot_recs_lm_pivot = None
        self.playlist_names = None

    def fit(self, df_pipeline):
        start = time.time()
        self.df_pipeline = df_pipeline
        self.least_misery_list = self.get_group_rec(df_pipeline)
        lm = time.time()
        print 'Least misery list calculated in: ', lm - start

        self.cluster()
        cl = time.time()
        print 'Clustering completed in: ', cl - lm

        self.recommend_playlists()
        self.playlist_recommender.print_playlists_withseeds(self.playlist_seed_names, self.playlist_names, self.playlist_seed_scores)
        print self.playlist_seed_df
        end = time.time()
        print 'Total time to completion: ', end - start


    def get_group_rec(self, df_pipeline):

        '''
        INPUT: list of user spotify ids
        OUTPUT: list of artists that cause least misery to all users
        '''

        gr = GroupRecommender()
        gr.load_model(self.model, model_cols = self.model_cols, user_col = self.user_col, 
            item_col = self.item_col, listen_col = self.listen_col)
        tot_recs = gr.create_user_rec_from_df(self.df_pipeline)

        

        tot_recs_df = tot_recs.to_dataframe()
        self.tot_recs_df = tot_recs_df # should remove
        least_misery_list, top_item_scores, df_least_misery, df_lm_allusers = gr.least_misery_list(tot_recs_df.copy())
        self.df_lm_allusers = df_lm_allusers
        self.df_least_misery = df_least_misery
        self.gr = gr
        self.score_col = gr.score_col #should be rank
        return least_misery_list

        '''
        # getting all user scores for each item in df least misery
        tot_recs_df_lm = tot_recs_df[tot_recs_df[self.item_col].isin(least_misery_list)]

            # creating "score" proxy from rank by dividing all ranks by max rank in least misery and subtracting from 1
            # later should use score and just normalize
        max_rank = tot_recs_df_lm[self.score_col].max()
        tot_recs_df_lm[self.score_col] = tot_recs_df_lm[self.score_col].apply(lambda x: 1 - ((1.0*x)/max_rank))
        tot_recs_lm_pivot = tot_recs_df_lm.pivot(self.item_col, self.user_col, self.score_col)
        self.tot_recs_lm_pivot = tot_recs_lm_pivot


        # lm_recs_df = tot_recs[[self.item_col].isin(list(least_misery_list))]
        # lm_recs_df = lm_recs_df.pivot
        return least_misery_list
        '''


    def cluster(self):
        '''
        Creates artist clusters from artists that cause least misery, ranks the clusters by top 5 artists,
        and passes those top 5 artists to be playlist seeds
        '''
        #import artist_term_clustering as a #causes graphlab to crash because KMeans is imported
        # atc = ArtistTermCluster()
        # labels = atc.fit(list(least_misery_list))
        # df_clusters = atc.create_clusters(labels, least_misery_list, self.df_least_misery)
        # playlist_seeds, df_cluster_rank, cluster_dict = atc.get_playlist_seeds(df_clusters)
        # self.playlist_seeds = playlist_seeds
        # self.df_cluster_rank = df_cluster_rank

        ac = ArtistClusterAF()
        self.ac = ac
        self.playlist_seed_df, self.playlist_seeds, self.playlist_seed_names, self.playlist_seed_scores = ac.fit(self.df_lm_allusers)

    def recommend_playlists(self):
        '''
        Takes playlist seeds, sends to Echonest API, and returns recommended playlists
        '''
        pr = PlaylistRecommender()
        pr.fit_multiple(self.playlist_seeds)
        self.playlist_list = pr.playlist_list
        self.playlist_id_list = pr.playlist_id_list
        self.playlist_spotify_id_list = pr.playlist_spotify_id_list
        self.playlist_recommender = pr
        self.playlist_names = pr.playlist_names




if __name__ == '__main__':

    start = time.time()
    # my_id = '1248440864'
    # liza_id = '1299323226'
    # userlist = [my_id, liza_id]
    df_pipeline = pd.read_csv('liza_ben_df.csv')[['user','artist_id','play_count']]
    s = spotify_functions.SpotifyFunctionsPublic()
    #df_pipeline = s.fit(userlist)
    model = gl.load_model('artist_sim_model_triplets')
    pl = Pipeline(model, model_cols = ['user','artist_id','play_count'], 
        user_col = 'user', item_col = 'artist_id', listen_col = 'play_count')
    pl.fit(df_pipeline)
    end = time.time()
    print pl.playlist_spotify_id_list
    print 'total time: ', end - start


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




