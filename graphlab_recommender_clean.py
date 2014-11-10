import graphlab as gl

def load_data():
    train_file = 'http://s3.amazonaws.com/GraphLab-Datasets/millionsong/10000.txt'

    # The below will download a 118 MB file.
    sf = gl.SFrame.read_csv(train_file, header=False, delimiter='\t', verbose=False)
    sf.rename({'X1':'user_id', 'X2':'song_id', 'X3':'listen_count'}).show()

    (train_set, test_set) = sf.random_split(0.8, seed=1)

    songs = gl.SFrame.read_csv('http://s3.amazonaws.com/GraphLab-Datasets/millionsong/song_data.csv', verbose=False)
    artists = songs[['song_id', 'artist_name']]

    # join on artists
    full_train = train_set.join(artists, on='song_id', how='inner')
    artist_train_sf = full_train[['user_id','artist_name','listen_count']]

    # group up listen counts
    grouped_artist_train_sf = artist_train_sf.groupby(['user_id', 'artist_name'], gl.aggregate.SUM('listen_count'))
    grouped_artist_train_sf.rename({'Sum of listen_count':'listen_count'})

    return grouped_artist_train_sf

def create_recommender(grouped_artist_train_sf):
    