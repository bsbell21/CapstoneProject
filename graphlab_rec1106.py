import graphlab as gl
from pymongo import MongoClient

def load_data():
    sf = gl.SFrame.read_csv('catalog_artist_ranking_init.csv')
    return sf

def connect_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["similarity"]
    collection_j = db.jaccard_init
    collection_p = db.pearson_init
    collection_c = db.cosine_init
    return db, collection_j, collection_p, collection_c

def create_recommender(sf, similarity = 'cosine'):
    item_sim_model = gl.item_similarity_recommender.create(sf, user_id='catalog', 
        item_id = 'artist_name', target = 'ranked_play_count', 
        similarity_type=similarity, only_top_k=1000000) 
    return item_sim_model

def create_similarity_dict(sf, item_sim_recommender, item_col = 'artist_name', value_col = 'ranked_play_count'):
    result = {}
    items = sf[item_col].unique()
    for item in items:
        result[item] = {}
        # setting similarity to itself as 1
        result[item][item] = 1
        # finding similar items and inputting values into dictionary
        similar_items = item_sim_recommender.get_similar_items(gl.SArray(data = [item]), k = -1)
        for sim_item in similar_items:
            result[item][sim_item['similar']] = sim_item['score']
    return result

def create_similarity_dict_mongo(sf, item_sim_recommender, collection, item_col = 'artist_name', value_col = 'ranked_play_count'):
    #result = {}
    items = sf[item_col].unique()
    for item in items:
        result = {}
        # setting similarity to itself as 1
        result[item] = 1
        # finding similar items and inputting values into dictionary
        similar_items = item_sim_recommender.get_similar_items(gl.SArray(data = [item]), k = -1)
        for sim_item in similar_items:
            result[sim_item['similar']] = sim_item['score']
        collection.insert(result)
    print 'Success!'

def create_similarity_dict_textdump(sf, item_sim_recommender, collection, item_col = 'artist_name', value_col = 'ranked_play_count'):
    items = sf[item_col].unique()
    for item in items:
        result = {}
        # setting similarity to itself as 1
        result[item] = 1
        # finding similar items and inputting values into dictionary
        similar_items = item_sim_recommender.get_similar_items(gl.SArray(data = [item]), k = -1)
        for sim_item in similar_items:
            result[sim_item['similar']] = sim_item['score']
        collection.insert(result)
    print 'Success!'


if __name__ == '__main__':
    sf = load_data()
    # db, collection_j, collection_p, collection_c = connect_mongo()

    # item_sim_model_j = create_recommender(sf, similarity = 'jaccard')
    # create_similarity_dict_mongo(sf, item_sim_model_j, collection_j)
    # item_sim_model_p = create_recommender(sf, similarity = 'pearson')
    # create_similarity_dict_mongo(sf, item_sim_model_p, collection_p)
    # item_sim_model_c = create_recommender(sf, similarity = 'cosine')
    # create_similarity_dict_mongo(sf, item_sim_model_c, collection_c)
    item_sim_model = create_recommender(sf, similarity = 'jaccard')
    artist_sim_dict = create_similarity_dict(sf, item_sim_model)
    f = open("jacard_init.txt", "wb")
    f.write(str(artist_sim_dict))
    f.close()
