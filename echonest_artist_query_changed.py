import requests
from pymongo import MongoClient
from pyechonest import config
config.ECHO_NEST_API_KEY="IHQR7II9KIYTTAPKS"
from pyechonest import artist
import pyen
en = pyen.Pyen("IHQR7II9KIYTTAPKS")
import graphlab as gl
import time

def load_data():
    sf = gl.SFrame.read_csv('catalog_artist_ranking_init.csv')
    artists = sf['artist_name'].unique()
    return artists

def connect_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client['echonest_artist']
    collection = db.artistdata
    return db, collection

def insert_artist_data(artist_name, collection):
    dic = en.get('artist/similar', name=artist_name, results = 100)
    dic['genres'] = en.get('artist/profile', name=artist_name, bucket = 'genre')['artist']['genres']
    collection.insert(dic)
    return dic['artists']

def insert_all_artists(artist_list, collection):
    new_artists = []
    new_artists_id = []
    for artist in artist_list:
        successful = False
        while not successful:
            try:
                sim_artists = insert_artist_data(artist, collection)
                for i in sim_artists:
                    if i['name'] not in artist_list:
                        new_artists.append(i['name'])
                        new_artists_id.append(i['id'])
                successful = True
            except:
                time.sleep(10)
                successful = False

    new_artists2, new_artists_id2 = insert_all_artists(new_artists, collection)
    new_artists.append(new_artists2)
    new_artists_id.append(new_artists_id2)
    return new_artists, new_artists_id

if __name__ == '__main__':
    db, collection = connect_mongo()
    artists = load_data()
    #dic = en.get('artist/similar', name='Chance the Rapper', results = 100)
    #dic['genres'] = en.get('artist/profile', name='Chance the Rapper', bucket = 'genre')['artist']['genres']
    new_artists, new_artists_id = insert_all_artists(artists, collection)
    # write new artists to file
    f = open("new_artists.txt", "wb")
    f.write(str(new_artists))
    f.close()
    f2 = open("new_artists_id.txt", "wb")
    f2.write(str(new_artists_id))
    f2.close()

    # ctr = artist.Artist('Chance the Rapper')
    # print "Artists similar to: %s:" % (ctr.name,)
    # for similar_artist in ctr.similar: print "\t%s" % (similar_artist.name,)