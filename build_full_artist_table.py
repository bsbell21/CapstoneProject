import graphlab as gl

def load_data():
    # artist term table
    artist_term = gl.SFrame.read_csv('msd_data/artist_term.csv', header = True)
    artist_term['play_count'] = 20
    
    artist_term = artist_term[['term', 'artist_id', 'play_count']]
    artist_term.rename({'term': 'user'})

    artist_term['table'] = 'artist_term'
    print artist_term[:10]

    # artist mbtag table
    artist_mbtag = gl.SFrame.read_csv('msd_data/artist_mbtag.csv', header = True)
    artist_mbtag['play_count'] = 20
    
    artist_mbtag = artist_mbtag[['mbtag', 'artist_id', 'play_count']]
    artist_mbtag.rename({'mbtag': 'user'})
    artist_mbtag['table'] = 'artist_mbtag'
    print artist_mbtag[:10]

    # artist similarity
    artist_similarity = gl.SFrame.read_csv('msd_data/artist_similarity.csv', header = True)
    artist_similarity['play_count'] = 40
    
    artist_similarity = artist_similarity[['target', 'similar', 'play_count']]
    artist_similarity.rename({'target':'user', 'similar':'artist_id'})
    artist_similarity['table'] = 'artist_similarity'
    print artist_similarity[:10]

    #triplets table
    artist_sf = gl.SFrame.read_csv('triplets_artists.csv', header = True)
    artist_sf['table'] = 'triplets_artists'

    #making full table
    full_sf = artist_sf.append(artist_term)
    full_sf = full_sf.append(artist_mbtag)
    full_sf = full_sf.append(artist_similarity)
    return full_sf


def run():
    full_sf = load_data()
    full_sf.save('full_artist_data.csv')
    full_sf.save('full_artist_data.csv', format = 'csv')
    full_sf.save('full_artist_data.csv', format = 'binary')

if __name__ == '__main__':
    run()