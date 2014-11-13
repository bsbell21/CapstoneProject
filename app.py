from flask import Flask
from flask import request
from flask import render_template
import requests
import bs4
import random
import pipeline_full_111214 as p

import numpy as np
import ipdb

'''
1. Create an app.py file in your my_app folder
* MyProject/my_app/app.py 1. Build a simple web homepage using flask. 
2. Once you have setup a working homepage... 
3. Build a submission_page that has an html form for the user 
to submit new text data. 4. Build a predict_page that processes 
the user submitted form data, and returns the result of your prediction.

'''
# load pickled models

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/group_login')
def group_login():
    return render_template('/group_login.html')

@app.route('/playlists', methods = ['POST'])
def playlists():
    # getting user names from group_login page
    user_names = []
    print 'in playlists'
    for i in range(5):
        user_name = request.form['user_' + str(i+1)]
        if len(user_name) > 0:
            user_names.append(user_name)
            print user_name

    print user_names
    pipeline = p.Pipeline('artist_sim_model_triplets', model_cols = ['user','artist_id','play_count'], 
        user_col = 'user', item_col = 'artist_id', listen_col = 'play_count')
    pipeline.fit(user_names)
    data = [str(i) for i in pipeline.playlist_list]
    return render_template('/playlists.html', data = data)







@app.route('/get_wallpaper', methods=['POST'])
def get_wallpaper():
    search = request.form['Search_Query']
    numrows = int(request.form['N_Rows'])
    numcols = int(request.form['N_Cols'])
    search_q = 'http://imgur.com/search?q=' + search
    data = requests.get(search_q)
    soup = bs4.BeautifulSoup(data.text)
    a = soup.findAll('a', {'class' : 'image-list-link'})

    # get image links and put in list
    images = []
    for i in a:
        images.append('http://' + i.findChildren('img')[0]['src'][2:])

    data = random.sample(images, numrows*numcols)
    print data
    return render_template('/get_wallpaper.html', data = data)




# @app.route('/submission_page')
# def submission_page():
#     return '''
#     <form action="/predict" method='POST' >
#         <input type="text" name="user_input" />
#         <input type="submit" />
#     </form>
#     '''

# @app.route('/predict', methods=['POST'] )

# create function to predict based off of pickled models

# def predict():
#     data = request.form['user_input']
#    # print data
#     #print type(data)
#     data = str(data)
#     #ipdb.set_trace()
#     vec = tf.transform([data]).toarray()
#     x = mnb.predict(vec)
#     return str(x)

# @app.route('/word_counter', methods=['POST'] )
# def word_counter():
#     # get data from request form, the key is the name you set in your form
#     data = request.form['user_input']

#     # convert data from unicode to string
#     data = str(data)

#     # run a simple program that counts all the words
#     dict_counter = {}
#     for word in data.lower().split():
#         if word not in dict_counter:
#             dict_counter[word] = 1
#         else:
#             dict_counter[word] += 1
#     total_words = len(dict_counter)

#     # now return your results
#     return 'Total words is %i, <br> dict_counter is: %s' % (total_words, dict_counter)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
