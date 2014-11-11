#import graphlab as gl
import pandas as pd
import numpy as np
import msd_sql_functions as msd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

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