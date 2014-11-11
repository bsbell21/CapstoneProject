import graphlab as gl

def load_data():
    artist_term = gl.SFrame.read_csv('triplets_artists.csv', header = True)
    return artist_term

def create_recommender(artist_term):
    item_sim_model = gl.item_similarity_recommender.create(artist_term, 'user', 'artist_id')
    return item_sim_model

def run():
    artist_term = load_data()
    item_sim_model = create_recommender(artist_term)
    item_sim_model.save('artist_sim_model_triplets')

if __name__ == '__main__':
    run()