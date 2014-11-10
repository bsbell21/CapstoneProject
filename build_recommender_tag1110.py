import graphlab as gl

def load_data():
    artist_term = gl.SFrame.read_csv('msd_data/artist_term.csv', header = True)
    return artist_term

def create_recommender(artist_term):
    item_sim_model_artistterms = gl.item_similarity_recommender.create(artist_term, 'term', 'artist_id')
    return item_sim_model_artistterms
    

def run():
    artist_term = load_data()
    item_sim_model_artistterms = create_recommender(artist_term)
    item_sim_model_artistterms.save('artist_sim_model_terms')

if __name__ == '__main__':
    run()