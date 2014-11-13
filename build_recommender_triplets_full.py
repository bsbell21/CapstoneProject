import graphlab as gl

def load_data():
    artist_sf = gl.SFrame.read_csv('triplets_artists.csv', header = True)
    return artist_sf

def create_recommender(artist_sf):
    item_sim_model = gl.item_similarity_recommender.create(artist_sf, 'user', 'artist_id', 
        only_top_k = 1000000)
    return item_sim_model

def run():
    artist_sf = load_data()
    item_sim_model = create_recommender(artist_sf)
    item_sim_model.save('artist_sim_model_triplets_full')

if __name__ == '__main__':
    run()