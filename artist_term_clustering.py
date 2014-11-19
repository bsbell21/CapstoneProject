#import graphlab as gl
import pandas as pd
import numpy as np
import msd_sql_functions as msd
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.feature_extraction.text import TfidfVectorizer
import ipdb



class ArtistClusterAF:
    def __init__(self):
        self.m = msd.MSD_Queries()
        self.labels = None
        self.artist_ids = None

        self.af = None

    def fit(self, df_least_misery, item_col = 'artist_id', score_col = 'rank', name_col = 'artist_name'):
        '''
        INPUT: DataFrame with names, ids, and least misery ranking for each artist
        OUTPUT: playlist seeds with rankings
        '''
        self.df_cols = df_least_misery.columns
        self.item_col = item_col
        self.score_col = score_col
        self.name_col = name_col
        self.artist_ids = list(df_least_misery[item_col].values)
        self.df_least_misery = df_least_misery

        term_docs = self.get_terms(self.artist_ids)
        feat_mtx = self.vectorize(term_docs)
        df_least_misery_clustered = ac.cluster(feat_mtx)
        self.playlist_seed_df, playlists_seeds = ac.get_playlist_seeds(df_least_misery_clustered)


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
        terms_all = self.m.gen_query(select_columns = 'artist_id, term', table = 'cluster_artist_nontriplets', 
            filter_column = 'artist_id', filter_values = artist_ids)
        df_terms = pd.DataFrame(terms_all)
        df_terms.columns = ['artist_id', 'term']
        # may have to get unique values here if not all artist ids guaranteed in databse
        term_docs = df_terms.groupby('artist_id')['term'].agg(' '.join).values
        # for artist_id in artist_ids:
        #     #print terms
        #     terms = df_terms[df_terms['artist_id'] == artist_id]
        #     if len(terms) > 0:
        #         terms_arr = terms['']
        #         terms_list = [i.replace(' ', '_') for i in terms_arr]
        #         doc = ' '.join(terms_list)
        #         term_docs.append(doc)
        #     else:
        #         term_docs.append('no_terms')
        
        return term_docs

    def vectorize(self, term_docs):
        '''
        clustering artists using affinity propogation, returns cluster labels
        '''
        tf = TfidfVectorizer()
        feat_mtx = tf.fit_transform(term_docs)
        return feat_mtx

    def cluster(self, feat_mtx):
        # clustering artists based on AffinityPropogation
        af = AffinityPropagation()
        af.fit(feat_mtx)
        self.labels = af.labels_
        self.af = af

        # adding cluster labels to least misery dataframe and sorting by rank and cluster
        df_least_misery_clustered = self.df_least_misery.copy()
        df_least_misery_clustered['cluster'] = self.labels
        df_least_misery_clustered[['cluster', self.score_col]] = df_least_misery_clustered[['cluster', self.score_col]].astype(float)
        ''' will do different sorting if not using rank '''
        df_least_misery_clustered = df_least_misery_clustered.sort(['cluster', self.score_col])
        return df_least_misery_clustered

    def get_playlist_seeds(self, df_least_misery_clustered, penalize_less_than = 4, penalization = 100):

        # taking top 5 artists per cluster and making playlist seeds
        playlist_seed_df_list = []
        avg_rank = []
        for i in np.unique(self.labels):
            top_cluster = df_least_misery_clustered[df_least_misery_clustered['cluster'] == i].head(5)
            rankmean = top_cluster[self.score_col].mean()
            # trying to remove clusters with < 4 artists from top
            if len(top_cluster) < penalize_less_than:
                rankmean += penalization
            top_cluster['avg_rank'] = rankmean
            avg_rank.append(rankmean)
            playlist_seed_df_list.append(top_cluster)

        # put together playlist seed df
        idx = np.argsort(np.array(avg_rank))
        playlist_seed_df_list = np.array(playlist_seed_df_list)[idx]

        playlist_seeds = [i[self.item_col].values for i in playlist_seed_df_list]
        playlist_seed_df = pd.concat(list(playlist_seed_df_list))

        return playlist_seed_df, playlist_seeds


class ArtistTermClusterTest:
    '''
    INPUT: list of artist_ids
    OUTPUT: cluster labels for each artist
    '''
    def __init__(self):
        self.m = msd.MSD_Queries()
        self.labels = None
        self.artist_ids = None
        self.n_clusters = None
        self.km = None

    def fit(self, artist_ids, n_clusters = 8):
        self.artist_ids = artist_ids
        term_docs = self.get_terms(artist_ids)
        labels = self.vectorize(term_docs)
        self.n_clusters = n_clusters
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
            terms = self.m.gen_query(select_columns = 'artist_id, term', table = 'cluster_artist_nontriplets', 
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

    def get_terms_quick(self, artist_ids):
        '''
        INPUT: list of artist_ids
        OUTPUT: dataframe with artist_ids and get_terms
        '''
        self.artist_ids = artist_ids
        if type(artist_ids) != list:
            artist_ids = [artist_ids]
        term_docs = []
        terms_all = self.m.gen_query(select_columns = 'artist_id, term', table = 'cluster_artist_nontriplets', 
            filter_column = 'artist_id', filter_values = artist_ids)
        df_terms = pd.DataFrame(terms_all)
        df_terms.columns = ['artist_id', 'term']
        # may have to get unique values here if not all artist ids guaranteed in databse
        term_docs = df_terms.groupby('artist_id')['term'].agg(' '.join).values
        # for artist_id in artist_ids:
        #     #print terms
        #     terms = df_terms[df_terms['artist_id'] == artist_id]
        #     if len(terms) > 0:
        #         terms_arr = terms['']
        #         terms_list = [i.replace(' ', '_') for i in terms_arr]
        #         doc = ' '.join(terms_list)
        #         term_docs.append(doc)
        #     else:
        #         term_docs.append('no_terms')
        
        return term_docs

    def vectorize(self, term_docs, n_clusters = 8):

        self.n_clusters = n_clusters
        tf = TfidfVectorizer()
        X = tf.fit_transform(term_docs)

        km = KMeans(n_clusters = n_clusters)

        artist_distance = km.fit_transform(X)
        #ipdb.set_trace()
        self.labels = km.labels_
        self.km = km
        return km.labels_, artist_distance

    def closest_playlist_seeds(self, artist_distance, num_closest = 5):
        '''
        INPUT: feature matrix, distance of each artist from centroid, number per cluster
        OUTPUT: 5 closest artists to each centroid, to use as seeds
        '''
        seeds = []
        for i in range(self.n_clusters):
            closest = np.argsort(artist_distance.T[i])[:num_closest]
            seeds.append(artist_ids[closest])
        return seeds

    def vectorize_af(self, term_docs):
        '''
        clustering artists using affinity propogation, returns cluster labels
        '''
        self.n_clusters = n_clusters
        tf = TfidfVectorizer()
        X = tf.fit_transform(term_docs)

        af = AffinityPropogation()
        af.fit(X)
        self.labels = af.labels_
        self.af = af
        return af.labels_

    def create_clusters_af(self, labels, artist_ids, item_col = 'artist_id', score_col = 'rank', 
        cluster_col = 'cluster', df_cols = ['artist_id', 'artist_name', 'cluster']):
        '''
        INPUT: 
        - list of artist_ids to cluster
        - df with clusters for each artist
        - 
        '''
        artist_names = np.array(m.get_artist_names(artist_ids))
        labels = np.array(labels).reshape(-1,1)
        artist_ids = np.array(artist_ids).reshape(-1,1)
        combo = np.hstack((artist_names, labels))
        df = pd.DataFrame(combo)
        df.columns = df_cols
    #     df[[cluster_col, score_col]] = df[[cluster_col, score_col]].astype(float)
        df = df.sort([cluster_col])
        return df

    def rank_clusters_af(self, df_clusters, least_misery_ranked_df, item_col = 'artist_id', score_col = 'rank', 
        cluster_col = 'cluster', df_cols = ['artist_id', 'artist_name', 'cluster']):

        '''
        INPUT: artists-cluster df
        OUTPUT: df witha artist-cluster and scores for each artist
        '''
        df = df.merge(least_misery_ranked_df, how = 'inner', on = item_col)
        df[[cluster_col, score_col]] = df[[cluster_col, score_col]].astype(float)
        df = df.sort([cluster_col, score_col])




    def vectorize_spectral(self, term_docs):
        pass


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

    def create_unranked_clusters(self, labels, artist_ids, item_col = 'artist_id', score_col = 'rank',
        cluster_col = 'cluster', df_cols = ['artist_id', 'artist_name', 'cluster']):
            artist_names = np.array(self.m.get_artist_names(artist_ids))
            labels = np.array(labels).reshape(-1,1)
            artist_ids = np.array(artist_ids).reshape(-1,1)
            combo = np.hstack((artist_names, labels))
            df = pd.DataFrame(combo)
            df.columns = df_cols
            df = df.sort([cluster_col])
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

class ArtistTermClusterLarge:
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
            # could do this more efficiently if queried all at once and then narrowed down
            terms = self.m.gen_query(select_columns = '"user", artist_id, play_count', table = 'full_artist_data', 
                filter_column = 'artist_id', filter_values = artist_id)
            #print terms
            if len(terms) > 0:
                # going through each term and adding it a number of times equal to the play count
                arr = np.array(terms)
                terms_arr = arr[:,0]
                terms_arr_count = arr[:,2]
                terms_list = []
                for idx, term in enumerate(terms_arr):
                    for i in range(terms_arr_count[idx]):
                        terms_list.append(term.replace(' ', '_'))
                #terms_list = [i.replace(' ', '_') for i in terms_arr]
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

        pass



if __name__ == '__main__':
    a = ArtistTermCluster()
    print a.get_terms('ARC8CQZ1187B98DECA')