#weTun.es

### A platform for group music discovery


Having a dinner party with your friends but don’t know what to play? weTun.es will take Spotify IDs from group members and generates a series of playlists that everyone will enjoy.

### The Challenge
- Taking user listen data from a group
- Recommending multiple internally consistent playlists for user’s to choose from

### The Data

- Million Song Database
	- Metadata includes:
	- artist tags (eg. Method Man: hip-hop, 90s)
	- similar artists (eg. Method Man: Redman)
- Taste Profiles
	- 1MM user-song connections
- Spotify public playlists
	- → “Implicit” user listen data
	
### How it works

#### Step 1: Preprocessing
- weTun.es recommendation engine is built off of the user-listen data in the Taste Profiles by creating an artist-artist similarity matrix to be used for collaborative filtering

#### Step 2: Creating a group session
- When users sign in to the weTun.es home page they create a group session by entering the Spotify IDs for every member of the group

#### Step 3: Individual Artist Recommendation
- weTun.es queries the Spotify API to get all of the public playlists for each user
- weTun.es counts artist appearances and considers each artist appearance to be one 'play' for a song of that artist
- once this 'implicit listen data' is created for each user, it is fed through the collaborative filter to get a list of preference scores for each user for each artist

#### Step 4: Group Artist Recommendation
- From the individual lists of user preferences, weTun.es takes a list of the top 200 artists for each user
- weTun.es then creates a list of group preferences using 'Least Misery' - assigning each artist the lowest preference score it recieved from any member of the group

#### Step 5: Clustering

- Once the group list is created, Affinity Propogation is implemented to group artists together with those that are most similar
- Affinity propogation, as opposed to K-Means, clusters solely based on similarity and therefore will make an appropriate number of clusters dependent upon the level of similarity between the preferences of the group members

#### Step 6: Playlists for All!
- Once the clusters are created, the top 5 artists from each cluster are taken as 'playlist seeds'
- the groups of playlists seeds are then ranked by average user preference for those 5 artists
- weTun.es queries the Echonest API to create playlists based on the seed artists for each group of playlist seeds
- weTun.es receives back a list of songs for each playlist, and creates the playlist in Spotify through the Spotify API
- weTun.es then renders the Spotify playlists on its site for the group to listen to and enjoy!

##### Important Files
- pipeline_full_131214.py - steps 3-6
- spotify_functions_mult131114 - querying Spotify API






