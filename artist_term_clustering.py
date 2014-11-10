#import graphlab as gl
import pandas as pd
import numpy as np
import msd_sql_functions as msd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class ArtistTermCluster:
    def __init__(self):
        self.m = msd.MSD_Queries()
        self.labels = None
        self.artist_ids = None
        self.n_clusters = None



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
            print terms
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

    def check_clusters(self):
        for idx in range(n_clusters):
            print 'cluster %d' % idx
            print self.artist_ids


    def pivot_table(self):
        '''
        INPUT: dataframe with artist_ids
        OUTPUT: pivoted feature featrix, and artist list
        '''

        pass

    def cluster(self):

        '''
        INPUT: 
        OUTPUT: 
        '''
        pass

if __name__ == '__main__':
    a = ArtistTermCluster()
    print a.get_terms('ARC8CQZ1187B98DECA')