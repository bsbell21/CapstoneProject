import psycopg2
from pymongo import MongoClient
import json

def connect_sql():
    conn = psycopg2.connect(dbname='msd', user='postgres', host='/tmp')
    c = conn.cursor()
    return conn, c

def connect_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["onefiftyk"]
    collection = db.userlistendata
    return db, collection

def check_sql(c):
    print c.fetchOne()

def insert_one(song_dic, collection, c, conn):
    catalog = song_dic['foreign_id'].split(':')[0].encode('ascii', 'ignore')
    artist_name = song_dic.get('artist_name', 'None').replace("'", "''").encode('ascii', 'ignore')
    artist_id = song_dic['artist_id'].encode('ascii', 'ignore')
    song_name = song_dic.get('song_name', 'None').replace("'", "''").encode('ascii', 'ignore')
    song_id = song_dic.get('song_id', 'None').encode('ascii', 'ignore')
    play_count = song_dic.get('play_count', 0)
    # for i in (catalog, artist_name, artist_id, song_name, song_id, play_count):
    #     if len(str(i)) < 1:
    #         i = 0

    #print (catalog, artist_name, artist_id, song_name, song_id, play_count)
    c.execute(
        '''
        INSERT INTO catalog_data VALUES
        ('%s', '%s', '%s', '%s', '%s', %d)
        '''
        % (catalog, artist_name, artist_id, song_name, song_id, play_count)
        )
    conn.commit()
#INSERT INTO catalog_data_init VALUES (u'CAXQEVN1332EA6DA56', u'Toto Cutugno', u'ARDHTLU1187B990EBA', u'LItaliano', u'SOJBCJA12A8C13AFBC', 1)
#INSERT INTO catalog_data_init VALUES('CAXQEVN1332EA6DA56', 'Toto Cutugno', 'ARDHTLU1187B990EBA', 'LItaliano', 'SOJBCJA12A8C13AFBC', 1)

def insert_all(collection, c, conn):
    errors = []
    for song_dic in collection.find():
        insert_one(song_dic, collection, c, conn)

        # # try: 
        # #     insert_one(song_dic, collection, c, conn)
        # # except Exception as e:
        # #     print e.message
        # #     errors.append(song_dic)
        #     continue

def create_artist_dic_vectors(user_name = 'catalog', artist_col_name = 'artist_name', play_count_name = 'play_count'):
    pass


if __name__ == '__main__':
    conn, c = connect_sql()
    db, collection = connect_mongo()
    errors = insert_all(collection, c, conn)
    print errors
    json.dumps(errors, open('errors.txt', 'wb'))

    conn.close()
    # c.execute('SELECT * FROM catalog_data_init')
    # print c.fetchone()

'''
SQL Queries:

SELECT catalog, artist_name, SUM(play_count) as play_count
INTO catalog_artist_count
FROM catalog_data
GROUP BY catalog, artist_name;
--> would actually want to use artist_id, but artist_name and id didn't align
--> TODO figure out later

basic normalization

SELECT catalog, SUM(play_count) as play_count
INTO catalog_count
FROM catalog_artist_count
GROUP BY catalog;

WITH temp AS (
SELECT cac.*, cc.play_count AS cat_play_count
FROM catalog_artist_count AS cac
LEFT JOIN catalog_count AS cc
ON cc.catalog = cac.catalog
)
SELECT *, play_count/cat_play_count AS norm_play_count
INTO catalog_artist_normcount
FROM temp
WHERE cat_play_count > 0;

--- create ranked ordering for each user with window function

SELECT *,
SUM(norm_play_count) OVER(PARTITION BY catalog ORDER BY play_count) AS ranked_play_count
INTO catalog_artist_ranking
FROM catalog_artist_normcount
WHERE play_count > 0;

--- make pivot table

CREATE EXTENSION tablefunc;

SELECT * 
INTO artist_pivot
FROM crosstab(
    'SELECT catalog, artist_name, ranked_play_count
    FROM catalog_artist_ranking')

WITH small AS (SELECT * FROM catalog_artist_ranking LIMIT 30)
SELECT * FROM crosstab('SELECT catalog, artist_name, ranked_play_count FROM small')
AS small(catalog varchar(250), artist_name varchar(250), ranked_play_count float);

WITH small AS (SELECT * FROM catalog_artist_ranking LIMIT 30)
SELECT * FROM small;

SELECT * INTO small FROM catalog_artist_ranking LIMIT 30;



