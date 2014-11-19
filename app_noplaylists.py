from flask import Flask
from flask import request
from flask import render_template
import requests
import random
import spotify_functions_mult131114 as spotify_functions
import pipeline_full_131214 as p
import time
import pandas as pd
import numpy as np
import ipdb
import graphlab as gl # have in final
import pickle
import re

'''
1. Create an app.py file in your my_app folder
* MyProject/my_app/app.py 1. Build a simple web homepage using flask. 
2. Once you have setup a working homepage... 
3. Build a submission_page that has an html form for the user 
to submit new text data. 4. Build a predict_page that processes 
the user submitted form data, and returns the result of your prediction.

'''


app = Flask(__name__)

# load model and create SpotifyFunctions instance to call later
s = spotify_functions.SpotifyFunctionsPublic()
model = gl.load_model('artist_sim_model_triplets') # have in final
# df_preload = pd.read_csv('liza_ben_df.csv')[['user','artist_id','play_count']] # remove from final


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign_in')
def sign_in():
    return render_template()

@app.route('/group_login_signin', methods = ['GET'])
def group_login_signin():
    # for user coming from signin page with access token
    access_token = request.values['access_token']
    refresh_token = request.values['refresh_token']
    display_name = request.values['display_name']
    user_id = request.values['user_id']
    data = [access_token, refresh_token, user_id, display_name]
    # ipdb.set_trace()
    return render_template('/group_login_signin.html', data = data)

@app.route('/group_login_quickstart')
def group_login():
    # for user coming from quick start
    return render_template('/group_login_quickstart.html')

@app.route('/playlists', methods = ['GET','POST'])
def playlists():
    s = spotify_functions.SpotifyFunctionsPublic()
    start = time.time()
    print 'in playlists'
    n_playlists = 5

    ''' Step 1: Getting private data from logged in user '''
    priv_check = False
    priv_check_count = 0
    if 'access_token' in request.form:
        access_token = request.form['access_token']
        priv_check_count += 1
    if 'refresh_token' in request.form:
        refresh_token = request.form['refresh_token']
        priv_check_count += 1
    if 'user_id' in request.form:
        prim_user_id = request.form['user_id']
        prim_user_id = prim_user_id.encode('ascii', 'ignore')
        priv_check_count += 1
    # if 'display_name' in request.form:
    #     display_name = request.form['display_name']
    #     priv_check += 1
    if priv_check_count > 2: # removing display name for now, would be > 3 with display name
        priv_check = True

    if priv_check: 
        spriv = spotify_functions.SpotifyFunctionsPrivate(access_token = access_token, refresh_token = refresh_token, user_id = prim_user_id)
        df_pipeline_private = spriv.fit()


    ''' Step 2. Getting public data from rest of group'''
    # getting user names from group_login page

    user_names = []

    for i in range(5):
        user_name = request.form['user_' + str(i+1)]
        if len(user_name) > 0:
            user_names.append(user_name)
            print user_name
    print user_names
    acceptable ='ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_-abcdefghijklmnopqrstuvwxyz1234567890?><:"{}+~ '
    user_names = [i.encode('ascii', 'ignore') for i in user_names]
    for i in user_names:
        for ch in i:
            if ch not in acceptable:
                i = i.replace(ch, '')
    df_pipeline = s.fit(user_names) # keep in final
    print 'user_names ', user_names

    # df_pipeline = df_preload # remove in final
    pipeline = p.Pipeline(model, model_cols = ['user','artist_id','play_count'], 
        user_col = 'user', item_col = 'artist_id', listen_col = 'play_count') # keep in final
    pipeline.fit(df_pipeline) # keep in final
    playlist_spotify_id_list, playlist_seed_names, playlist_seed_scores, playlist_names = pipeline.playlist_spotify_id_list, pipeline.playlist_seed_names, pipeline.playlist_seed_scores, pipeline.playlist_names # keep in final 
    
    # if preloading all data use below
    # playlist_spotify_id_list, playlist_seed_names, playlist_seed_scores, playlist_names = pickle.load(open('preloads/pipeline_objs_1116.pkl', 'rb')) # remove in final
    # pl_playlist_ids = ['3HoRrJNYuDkhATqm4NEZSW','0jXqizs7Z0sWWVaJaqyP2r'] # remove in final

    ''' Step 3. Creating Playlists '''
    all_data = []
    playlist_ids = []
    user_ids = list(playlist_seed_scores.columns - ['cluster','cluster_score'])
    print 'user ids ascii ', user_ids
    #user_ids = [i.decode('utf-8') for i in user_ids]
    print 'user ids utf-8 ', user_ids
    playlist_names = playlist_names[:n_playlists]
    embed_base = "https://embed.spotify.com/?uri=https://play.spotify.com/user/1248440864/playlist/"
    for idx, playlist in enumerate(playlist_spotify_id_list[:n_playlists]):
        all_data.append({})
        playlist_id = s.create_playlist(playlist, playlist_names[idx]) # keep in final
        # playlist_id = pl_playlist_ids[idx] # remove in final
        playlist_ids.append(playlist_id)
        p_id = playlist_id
        p_id = re.sub(r'([^\s\w]|_)+', '', p_id)
        all_data[idx]['playlist_id'] = p_id

        all_data[idx]['playlist_html'] = embed_base + str(playlist_id)
        all_data[idx]['seed_artist_names'] = playlist_seed_names[idx].decode('utf-8')
        # all_data[idx]['seed_artist_names'] = all_data[idx]['seed_artist_names'].encode('ascii','ignore')
        all_data[idx]['scores'] = {} # should remove this or userids/scores - redundant
        all_data[idx]['user_ids'] = []
        all_data[idx]['user_scores'] = []
        all_data[idx]['playlist_name'] = playlist_names[idx]
        for user_id in user_ids:
            all_data[idx]['scores'][user_id] = playlist_seed_scores[user_id][idx]
            # all_data[idx]['user_ids'].append(re.sub(r'([^\s\w]|_)+', '', user_isign_ind))
            all_data[idx]['user_ids'].append(user_id.encode('ascii', 'ignore'))
            all_data[idx]['user_scores'].append(round(playlist_seed_scores[user_id][idx],2))
        # passing in user_id list as string to avoid problems with translation
        all_data[idx]['user_ids'] = str(all_data[idx]['user_ids'])

    
    playlist_html = [embed_base + str(i) for i in playlist_ids]
    for i in all_data:
        print i
        print type(i)
    end = time.time()
    print 'total time through playlists: ', end - start
    print 'user_ids ', user_ids
    print 'playlist_html ', playlist_html
    print 'playlist_ids ', playlist_ids

    # data = playlist_html
    data = all_data
    for i in data:
        print i
        print type(i)
    end = time.time()
    print 'total time through playlists: ', end - start
    return render_template('/playlists.html', data = data)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
