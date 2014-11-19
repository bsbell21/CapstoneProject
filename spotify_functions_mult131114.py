import spotipy as sp
import spotipy.util as util
import pyen
import pandas as pd
import numpy as np
import msd_sql_functions as msd
import time
import ipdb
en = pyen.Pyen("IHQR7II9KIYTTAPKS")
# token = util.prompt_for_user_token('bsbell21', client_id = '530ddf60a0e840369395009076d9fde7', client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
# s = sp.Spotify(auth = token)


'''
Features to add on inbound data collection:
1. removing duplicates for artists that have more than 1 id
'''

class SpotifyFunctionsMain:
    def __init__(self):
        self.df_pipeline_pub = None
        self.df_private_pipeline = None
        self.df_pipeline = None
        self.spub = None

    def fit(self,  public_user_ids, private_users_dict_list = None, spotify_public_object = None):
        '''
        INPUT: LIST of user_ids for each user to get public data on
               LIST of dictionaries containing user_id, access_token, and refresh_token for each private user
               SpotifyFunctionsPublic object to use


        OUTPUT: DataFrame to be fed into pipeline
        '''
        df_pipeline_list = []

        # getting data for private users
        if private_users_dict_list != None:
            private_pipeline_list = []
            for dic in private_users_dict_list:
                spriv = SpotifyFunctionsPrivate(user_id = dic['user_id'], access_token = dic['access_token'], refresh_token = dic['refresh_token'])
                df_pipeline = spriv.fit()
                private_pipeline_list.append(df_pipeline)
            df_private_pipeline = pd.concat(private_pipeline_list)
            df_pipeline_list.append(df_private_pipeline)

        # creating df for all public users
        if spotify_public_object == None:
            spub = SpotifyFunctionsPublic()
        else:
            spub = spotify_public_object
        df_pipeline_pub = spub.fit(public_user_ids)
        df_pipeline_list.append(df_pipeline_pub)

        # creating single df_pipeline
        df_pipeline = pd.concat(df_pipeline_list)

        self.df_pipeline_pub = df_pipeline_pub
        self.df_private_pipeline = df_private_pipeline 
        self.df_pipeline = df_pipeline
        self.spub = spub
        return df_pipeline




class SpotifyFunctionsPrivate:
    def __init__(self, user_id, access_token, refresh_token):
        self.user_saved_tracks = None
        self.m = msd.MSD_Queries()
        self.artist_data_echonest = None
        self.user_id = user_id
        self.token = access_token
        self.refresh_token = refresh_token
        self.s = sp.Spotify(auth = self.token)

    def fit(self, update = False):
        user_id = self.user_id
        start = time.time()
        # checking for stored data
        private_stored_data_count = self.m.count_user_stored_private_data(user_id)
        if (private_stored_data_count > 0) and (update == False):
            df_pipeline = self.m.get_user_stored_private_data(user_id)
            print 'getting stored private data'

        # code to collect data and store
        else:
            user_saved_tracks = self.get_user_saved_tracks()
            saved_track_artist_data = self.get_saved_track_artist_data()

            saved_tracks_time = time.time()
            print 'Saved tracks collected in: ', saved_tracks_time - start

            user_playlists = self.get_user_playlists()
            playlist_artist_data = self.get_playlist_artist_data()

            playlists_time = time.time()
            print 'Playlists collected in: ', playlists_time - saved_tracks_time 

            artist_data = playlist_artist_data + saved_track_artist_data
            self.artist_data = artist_data
            df_pipeline = self.create_df_pipeline_from_artist_data(artist_data)
            self.df_pipeline = df_pipeline

            pipeline_time = time.time()
            print 'Pipeline created in: ', pipeline_time - playlists_time
            if update == True:
                self.m.delete_user_stored_data_private(user_id)
                print user_id, 'DATA DELETED'
            self.m.insert_dataframe(df_pipeline, 'stored_listen_data_private')

        end = time.time()
        print 'Total time to completion', end - start

        return df_pipeline


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
        return user_saved_tracks

    def get_saved_track_artist_data(self):
        artist_data = []
        for item in self.user_saved_tracks:
            dic = {}
            dic['artist_name'] = item['track']['artists'][0]['name']
            dic['artist_id_spotify']= item['track']['artists'][0]['id']
            artist_data.append(dic)
        self.saved_tracks_artist_data = artist_data
        return artist_data

    def get_user_playlists(self, lim = 50):
        '''
        INPUT: user_id
        OUTPUT: list of users public playlists
        '''
        user_id = self.user_id
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
        # adding in starred playlist
        starred_playlist = self.s.user_playlist(user_id)
        user_playlist_objects.append(starred_playlist)

        self.user_playlists = user_playlists
        self.user_playlist_objects = user_playlist_objects
        return user_playlists

    def get_playlist_artist_data(self):
        artist_data = []
        for playlist_data in self.user_playlist_objects:
            for item in playlist_data['tracks']['items']:
                dic = {}
                dic['artist_name'] = item['track']['artists'][0]['name']
                dic['artist_id_spotify']= item['track']['artists'][0]['id']
                artist_data.append(dic)

        self.playlist_artist_data = artist_data
        return artist_data

        


    def create_df_pipeline_from_artist_data(self, artist_data):
        '''
        INPUT: list of dictionaries for each artist containing 'artist_name' and 'artist_id_spotify' keys
        OUTPUT: DataFrame for pipeline with echonest_ids for artists in database
        '''
        user_id = self.user_id
        df_artist_data = pd.DataFrame(artist_data)
        df_artist_data['count'] = 1
        print 'artist data ', df_artist_data
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
        df_pipeline['user'] = user_id
        df_pipeline = df_pipeline[['user','artist_id','count']]
        df_pipeline.columns = ['user','artist_id','play_count']
        self.df_pipeline = df_pipeline

        return df_pipeline

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


class SpotifyFunctionsPublic:
    def __init__(self):
        self.user_saved_tracks = None
        self.m = msd.MSD_Queries()
        self.artist_data_echonest = None
        self.my_id = '1248440864' # will always use my id
        self.token = util.prompt_for_user_token(self.my_id, 
            scope = 'playlist-modify-public user-library-read playlist-read-private playlist-modify-private user-library-modify', client_id = '530ddf60a0e840369395009076d9fde7', 
            client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
        self.s = sp.Spotify(auth = self.token)

    def fit(self, user_ids, update = []):
        '''
        INPUT: list of user_ids, list of user_ids to update
        OUTPUT: DataFrame with listen data for all users' public playlists
        if update, will delete user data from database and repopulate it, else will look for data in database
        and use that if it exists
        '''
        #adding this as fix but want to remove
        self.token = util.prompt_for_user_token(self.my_id, 
            scope = 'playlist-modify-public user-library-read playlist-read-private playlist-modify-private user-library-modify', client_id = '530ddf60a0e840369395009076d9fde7', 
            client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
        # remove above when you figure out expiration

        df_pipeline_list = []
        for user_id in user_ids:
            if user_id in update:
                df_pipeline_user = self.fit_one(user_id, update = True)
            else:
                df_pipeline_user = self.fit_one(user_id)
            # stored_data_count = self.m.count_user_stored_data(user_id)
            # if (stored_data_count > 0) and (update == False):
            #     df_pipeline_user = self.m.get_user_stored_data(user_id)
            # else:
            #     df_pipeline_user = self.fit_one(user_id)
            #     if update == True:
            #         self.m.delete_user_stored_data(user_id)
            #     self.m.insert_dataframe(df_pipeline_user, 'stored_listen_data')
            # creating appropriate column structure
            # MOVING THIS TO GroupRecommender
            # if self.listen_col:
            #     df_pipeline_user.columns = [self.user_col, self.item_col, self.listen_col]
            # else:
            #     df_pipeline_user = df_pipeline_user[df_pipeline_user.columns[:2]]
            #     df_pipeline_user.columns = [self.user_col, self.item_col]

            # appending to df_pipeline_list to aggregate
            if len(df_pipeline_user) > 0:
                df_pipeline_list.append(df_pipeline_user)


        df_pipeline_full = pd.concat(df_pipeline_list)#.reset_index()
        return df_pipeline_full

    def fit_one(self, user_id, update = False):
        print user_id
        ''' 
        something really weird happening, number of artists collected from liza dropped
        from 400 to 308 with no changes in artist selection/deletion
        may have something to do with unicode? 
        Should double check getting all playlists
        Should double check that unicode items are not being removed ie /x etc
        moving on to creating full pipeline
        '''
        self.token = util.prompt_for_user_token(self.my_id, 
            scope = 'playlist-modify-public user-library-read playlist-read-private playlist-modify-private user-library-modify', client_id = '530ddf60a0e840369395009076d9fde7', 
            client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
        stored_data_count = self.m.count_user_stored_data(user_id)
        private_stored_data_count = self.m.count_user_stored_private_data(user_id)
        print 'stored_data_count, ', stored_data_count
        print 'username, ', user_id
        print 'update, ', update
        if (private_stored_data_count > 0) and (update == False):
            df_pipeline = self.m.get_user_stored_private_data(user_id)
            print 'getting stored private data'
        elif (stored_data_count > 0) and (update == False):
            df_pipeline = self.m.get_user_stored_data(user_id)
            print 'getting stored public data'
        else:
            'replacing data..'
            ''' OLD CODE '''
            user_playlists = self.get_user_public_playlists(user_id)
            if len(user_playlists) > 0:
                print len(user_playlists)
                df_pipeline, artist_data_echonest = self.get_playlist_data(user_id)
                print df_pipeline
                ''' END OF OLD CODE '''
                if update == True:
                    self.m.delete_user_stored_data(user_id)
                    print user_id, 'DATA DELETED'
                self.m.insert_dataframe(df_pipeline, 'stored_listen_data')
            else:
                df_pipeline = []
                print 'no data found for user ', user_id

        return df_pipeline
            


    def create_playlist(self, spotify_track_ids, playlist_name):
        self.token = util.prompt_for_user_token(self.my_id, 
            scope = 'playlist-modify-public user-library-read playlist-read-private playlist-modify-private user-library-modify', client_id = '530ddf60a0e840369395009076d9fde7', 
            client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
        #REMOVE ABOVE WHEN YOU GET IT NOT TO EXPIRE

        print 'spotify ids: ', spotify_track_ids 
        playlist_dic = self.s.user_playlist_create(self.my_id, playlist_name)
        playlist_id = str(playlist_dic['id'])
        results = self.s.user_playlist_add_tracks(self.my_id, playlist_id, spotify_track_ids)
        print results
        print 'playlist run through'
        print playlist_id
        print playlist_dic
        return playlist_id

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

    def get_playlist_data(self, user_id):
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
        print 'artist data ', df_artist_data
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
        df_pipeline['user'] = user_id
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


# class SpotifyFunctionsInd:
#     def __init__(self, access_token, refresh_token):
#         self.token = access_token
#         self.refresh_token = refresh_token
#         self.s = None
#         self.user_saved_tracks = None
#         self.user_id = None
#         self.m = msd.MSD_Queries()
#         self.artist_data_echonest = None

#     def fit(self, user_id):
#         ''' 
#         something really weird happening, number of artists collected from liza dropped
#         from 400 to 308 with no changes in artist selection/deletion
#         may have something to do with unicode? 
#         Should double check getting all playlists
#         Should double check that unicode items are not being removed ie /x etc
#         moving on to creating full pipeline
#         '''
#         self.user_id = user_id
#         # self.token = util.prompt_for_user_token(user_id, scope = 'user-library-read', client_id = '530ddf60a0e840369395009076d9fde7', 
#         #     client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')

#         print 'token created'
#         self.s = sp.Spotify(auth = self.token)

#         # new code start - getting playlist authentication
#         # self.token_playlist = util.prompt_for_user_token(user_id, scope = 'playlist-modify-public', client_id = '530ddf60a0e840369395009076d9fde7', 
#         #     client_secret = 'd1974e81df054fb2bffa895b741f96f6', redirect_uri = 'https://github.com/bsbell21')
#         # self.s_playlist = sp.Spotify(auth = self.token_playlist)
#         # self.s_playlist.trace = False
#         # new code end

#         self.get_user_saved_tracks() 
#         user_playlists = self.get_user_public_playlists(user_id)
#         df_pipeline, artist_data_echonest = self.get_playlist_data()
#         return df_pipeline


#     def create_playlist(self, spotify_track_ids, playlist_name):
#         playlists = self.s_playlist.user_playlist_create(self.user_id, playlist_name)

#     def convert_artist_id(self, spotify_id):
#         '''
#         INPUT: spotify artist id
#         OUTPUT: echonest artist id
#         '''
#         profile = en.get('artist/profile', id = 'spotify:artist:' + str(spotify_id))
#         echonest_id = profile['artist']['id']
#         return echonest_id

#     def get_user_saved_tracks(self, lim = 50):
#         '''
#         INPUT: none
#         OUTPUT: list of users saved tracks
#         '''
#         user_saved_tracks = []
#         user_saved_tracks_init = self.s.current_user_saved_tracks(limit = lim, offset = 0)
#         total_tracks = user_saved_tracks_init['total']

#         x = 0 # initializing offset

#         # looping through each iteration of saved tracks (given limit on query), to get all in user_saved_tracks
#         while x < total_tracks:
#             user_tracks = self.s.current_user_saved_tracks(limit = lim, offset = x)
#             for item in user_tracks['items']:
#                 user_saved_tracks.append(item)
#             x += lim
#         self.user_saved_tracks = user_saved_tracks


#     def get_user_public_playlists(self, user_id, lim = 50):
#         '''
#         INPUT: user_id
#         OUTPUT: list of users public playlists
#         '''
#         user_playlists = []
#         user_playlist_objects = []
#         user_playlists_init = self.s.user_playlists(user_id, limit = lim)
#         total_playlists = user_playlists_init['total']

#         x = 0 # initializing offset

#         while x < total_playlists:
#             user_playlists_tmp = self.s.user_playlists(user_id, limit = lim, offset = x)
#             for item in user_playlists_tmp['items']:
#                 user_playlists.append(item)
#                 playlist_id = item['id']
#                 owner = item['owner']['id']
#                 playlist_data = self.s.user_playlist(owner, playlist_id)
#                 user_playlist_objects.append(playlist_data)

#             x += lim

#         self.user_playlists = user_playlists
#         self.user_playlist_objects = user_playlist_objects
#         return user_playlists

#     def get_playlist_data(self):
#         '''
#         INPUT: all playlists
#         OUTPUT: df with artist id, name, and count


#         '''

#         artist_data = []
#         for playlist_data in self.user_playlist_objects:
#             # playlist_id = playlist_dic['id']
#             # owner = playlist_dic['owner']['id']
#             # playlist_data = self.s.user_playlist(owner, playlist_id)

#             for item in playlist_data['tracks']['items']:
#                 dic = {}
#                 dic['artist_name'] = item['track']['artists'][0]['name']
#                 dic['artist_id_spotify']= item['track']['artists'][0]['id']
#                 artist_data.append(dic)


#         df_artist_data = pd.DataFrame(artist_data)
#         df_artist_data['count'] = 1
#         df_artist_data = df_artist_data.groupby('artist_id_spotify').agg({'artist_name': min, 'count': np.sum}).reset_index()

#         # creating artist name column that will sync w sql database
#         df_artist_data['artist_name_sql'] = df_artist_data['artist_name'].apply(lambda x: 
#             x.encode('ascii','ignore').replace("'",'"'))

#         artist_ids = list(df_artist_data['artist_id_spotify'].values)
#         artist_names = list(df_artist_data['artist_name_sql'].values)

#         # getting echonest ids for each artist name from sql database
#         df_echonest_ids = self.get_echonest_ids_by_artistname(artist_names)

#         # joining echonest ids to artist data dataframe
#         df_artist_data_echonest = df_echonest_ids.merge(df_artist_data, how = 'left', on = 'artist_name')

#         # making sure no multiples coming of move
#         if len(df_artist_data_echonest) != len(df_echonest_ids):
#             print 'Warning! Merging caused duplication' #could aggregate by name and take max count but leave like this for now

#         '''
#         For artist names not found in database:
#         1. find artists not found in initial database search
#         2. query echonest api for ids
#         3. check ids against ids in database and remove all not in database
#         '''
#         # get all artists not found in database

#         #

#         self.artist_data_echonest = df_artist_data_echonest

#         # converting into format for pipeline
#         df_pipeline = df_artist_data_echonest[['artist_id', 'count']]
#         df_pipeline['user'] = self.user_id
#         df_pipeline = df_pipeline[['user','artist_id','count']]
#         df_pipeline.columns = ['user','artist_id','play_count']
#         self.df_pipeline = df_pipeline

#         return df_pipeline, df_artist_data_echonest

#     def convert_spotify_echonest_ids(self, artist_ids):

#         # convert to echonest ids
        
#         echonest_ids = []
#         for idx, i in enumerate(artist_ids):
#             try:
#                 echonest_ids.append(self.convert_artist_id(i))
#             except:
#                 echonest_ids.append('error')
#                 print 'error'

#         return echonest_ids

#     def get_echonest_ids_by_artistname(self, list_artist_names, unique = True):
#         '''
#         INPUT: list of artist names`
#         OUTPUT: echonest ids of artists from msd database
#         '''
#         list_artist_names_clean = [i.replace("'",'"') for i in list_artist_names]
#         echonest_ids = self.m.get_artist_ids(list_artist_names_clean)
#         df = pd.DataFrame(echonest_ids)
#         df.columns = ['artist_id', 'artist_name']
#         df['artist_name'] = df['artist_name'].apply(lambda x: x.encode('ascii', 'encode'))
#         if unique:
#             df['count'] = 1
#             df_agg = df.groupby(['artist_name']).agg({'artist_id': min, 'count': sum}).reset_index()
#             df_unique = df_agg[df_agg['count'] == 1]
#             df = df_unique[['artist_id', 'artist_name']]
#         return df





#     def get_artists_following(self, userid):
#         '''
#         INPUT: user id
#         OUTPUT: list of artist ids for artists the user is following
#         '''
#         pass




#     def get_user_artists_ids(self):
#         '''
#         INPUT: none
#         OUTPUT: echonest ids for each artist and play count
#         '''
#         total_tracks = self.user_saved_tracks['total']
#         for item in self.user_saved_tracks['items']:
#             track_info = {}
#             track_info['added_at'] = item['added_at']


if __name__ == '__main__':
    s = SpotifyFunctionsPublic()
    print s.fit(['1248440864'])
    pass



