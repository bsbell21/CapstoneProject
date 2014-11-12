import spotipy as sp
import spotipy.util as util
import pyen
import pandas as pd
import numpy as np
import msd_sql_functions as msd
en = pyen.Pyen("IHQR7II9KIYTTAPKS")
# token = util.prompt_for_user_token('bsbell21', client_id = '530ddf60a0e840369395009076d9fde7', client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
# s = sp.Spotify(auth = token)


'''
Features to add on inbound data collection:
1. removing duplicates for artists that have more than 1 id
'''

class SpotifyFunctions:
    def __init__(self):
        self.token = None
        self.s = None
        self.user_saved_tracks = None
        self.user_id = None
        self.m = msd.MSD_Queries()
        self.artist_data_echonest = None

    def fit(self, user_id):
        ''' 
        something really weird happening, number of artists collected from liza dropped
        from 400 to 308 with no changes in artist selection/deletion
        may have something to do with unicode? 
        Should double check getting all playlists
        Should double check that unicode items are not being removed ie /x etc
        moving on to creating full pipeline
        '''
        self.user_id = user_id
        self.token = util.prompt_for_user_token(user_id, scope = 'user-library-read', client_id = '530ddf60a0e840369395009076d9fde7', 
            client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
        print 'token created'
        self.s = sp.Spotify(auth = self.token)
        #self.get_user_saved_tracks()
        user_playlists = self.get_user_public_playlists(user_id)
        df_pipeline, artist_data_echonest = self.get_playlist_data()
        return df_pipeline




    def convert_artist_id(self, spotify_id):
        '''
        INPUT: spotify artist id
        OUTPUT: echonest artist id
        '''
        profile = en.get('artist/profile', id = 'spotify:artist:' + str(spotify_id))
        echonest_id = profile['artist']['id']
        return echonest_id

    def get_user_saved_tracks(self, lim = 50):
        '''
        INPUT: none
        OUTPUT: list of users saved tracks
        '''
        user_saved_tracks = []
        user_saved_tracks_init = self.s.current_user_saved_tracks(limit = lim, offset = 0)
        total_tracks = user_saved_tracks_init['total']

        x = 0 # initializing offset

        # looping through each iteration of saved tracks (given limit on query), to get all in user_saved_tracks
        while x < total_tracks:
            user_tracks = self.s.current_user_saved_tracks(limit = lim, offset = x)
            for item in user_tracks['items']:
                user_saved_tracks.append(item)
            x += lim
        self.user_saved_tracks = user_saved_tracks


    def get_user_public_playlists(self, user_id, lim = 50):
        '''
        INPUT: user_id
        OUTPUT: list of users public playlists
        '''
        user_playlists = []
        user_playlist_objects = []
        user_playlists_init = self.s.user_playlists(user_id, limit = lim)
        total_playlists = user_playlists_init['total']

        x = 0 # initializing offset

        while x < total_playlists:
            user_playlists_tmp = self.s.user_playlists(user_id, limit = lim, offset = x)
            for item in user_playlists_tmp['items']:
                user_playlists.append(item)
                playlist_id = item['id']
                owner = item['owner']['id']
                playlist_data = self.s.user_playlist(owner, playlist_id)
                user_playlist_objects.append(playlist_data)

            x += lim

        self.user_playlists = user_playlists
        self.user_playlist_objects = user_playlist_objects
        return user_playlists

    def get_playlist_data(self):
        '''
        INPUT: all playlists
        OUTPUT: df with artist id, name, and count
        '''
        artist_data = []
        for playlist_data in self.user_playlist_objects:
            # playlist_id = playlist_dic['id']
            # owner = playlist_dic['owner']['id']
            # playlist_data = self.s.user_playlist(owner, playlist_id)

            for item in playlist_data['tracks']['items']:
                dic = {}
                dic['artist_name'] = item['track']['artists'][0]['name']
                dic['artist_id_spotify']= item['track']['artists'][0]['id']
                artist_data.append(dic)


        df_artist_data = pd.DataFrame(artist_data)
        df_artist_data['count'] = 1
        df_artist_data = df_artist_data.groupby('artist_id_spotify').agg({'artist_name': min, 'count': np.sum}).reset_index()

        # creating artist name column that will sync w sql database
        df_artist_data['artist_name_sql'] = df_artist_data['artist_name'].apply(lambda x: 
            x.encode('ascii','ignore').replace("'",'"'))

        artist_ids = list(df_artist_data['artist_id_spotify'].values)
        artist_names = list(df_artist_data['artist_name_sql'].values)

        # getting echonest ids for each artist name from sql database
        df_echonest_ids = self.get_echonest_ids_by_artistname(artist_names)

        # joining echonest ids to artist data dataframe
        df_artist_data_echonest = df_echonest_ids.merge(df_artist_data, how = 'left', on = 'artist_name')

        # making sure no multiples coming of move
        if len(df_artist_data_echonest) != len(df_echonest_ids):
            print 'Warning! Merging caused duplication' #could aggregate by name and take max count but leave like this for now

        '''
        For artist names not found in database:
        1. find artists not found in initial database search
        2. query echonest api for ids
        3. check ids against ids in database and remove all not in database
        '''
        # get all artists not found in database

        #

        self.artist_data_echonest = df_artist_data_echonest

        # converting into format for pipeline
        df_pipeline = df_artist_data_echonest[['artist_id', 'count']]
        df_pipeline['user'] = self.user_id
        df_pipeline = df_pipeline[['user','artist_id','count']]
        df_pipeline.columns = ['user','artist_id','play_count']
        self.df_pipeline = df_pipeline

        return df_pipeline, df_artist_data_echonest

    def convert_spotify_echonest_ids(self, artist_ids):

        # convert to echonest ids
        
        echonest_ids = []
        for idx, i in enumerate(artist_ids):
            try:
                echonest_ids.append(self.convert_artist_id(i))
            except:
                echonest_ids.append('error')
                print 'error'

        return echonest_ids

    def get_echonest_ids_by_artistname(self, list_artist_names, unique = True):
        '''
        INPUT: list of artist names`
        OUTPUT: echonest ids of artists from msd database
        '''
        list_artist_names_clean = [i.replace("'",'"') for i in list_artist_names]
        echonest_ids = self.m.get_artist_ids(list_artist_names_clean)
        df = pd.DataFrame(echonest_ids)
        df.columns = ['artist_id', 'artist_name']
        df['artist_name'] = df['artist_name'].apply(lambda x: x.encode('ascii', 'encode'))
        if unique:
            df['count'] = 1
            df_agg = df.groupby(['artist_name']).agg({'artist_id': min, 'count': sum}).reset_index()
            df_unique = df_agg[df_agg['count'] == 1]
            df = df_unique[['artist_id', 'artist_name']]
        return df





    def get_artists_following(self, userid):
        '''
        INPUT: user id
        OUTPUT: list of artist ids for artists the user is following
        '''
        pass




    def get_user_artists_ids(self):
        '''
        INPUT: none
        OUTPUT: echonest ids for each artist and play count
        '''
        total_tracks = self.user_saved_tracks['total']
        for item in self.user_saved_tracks['items']:
            track_info = {}
            track_info['added_at'] = item['added_at']


if __name__ == '__main__':
    pass



