
import psycopg2
import numpy as np
import pandas as pd

class MSD_Queries:
    def __init__(self):
        self.conn = psycopg2.connect(dbname='msd', user='postgres', host='/tmp')
        self.c = self.conn.cursor()

    def delete_user_stored_data(self, user_id):
        '''
        INPUT: user_id
        OUTPUT: none - deletes users data from stored_listen_data
        '''
        self.c.execute(
                '''
                DELETE FROM stored_listen_data WHERE "user" = '%s';
                ''' % (str(user_id))
            )
        self.conn.commit()

    def delete_user_stored_data_private(self, user_id):
        '''
        INPUT: user_id
        OUTPUT: none - deletes users data from stored_listen_data
        '''
        self.c.execute(
                '''
                DELETE FROM stored_listen_data_private WHERE "user" = '%s';
                ''' % (str(user_id))
            )
        self.conn.commit()

    def insert_dataframe(self, df, table):
        # ensuring unique values in index
        df = df.reset_index().drop('index',axis = 1)

        for i in range(len(df)):
            # could potentially do this all in one command instead of looping
            self.c.execute(
                '''
                INSERT INTO %s VALUES %s;
                ''' % (str(table), str(tuple(df.ix[i].values)))
                )
            print 'INSERT INTO %s VALUES %s;' % (str(table), str(tuple(df.ix[i].values)))

        self.conn.commit()

    def get_user_stored_data(self,  user_id):
        '''
        INPUT: user_id
        OUTPUT: DataFrame of stored data for user
        '''
        user_data = self.gen_query('*', 'stored_listen_data', '"user"', user_id)
        df = pd.DataFrame(user_data)
        df.columns = ['user', 'artist_id', 'play_count']
        return df

    def count_user_stored_data(self, user_id):
        '''
        INPUT: user_id
        OUTPUT: int, count of stored data for user
        '''
        return int(self.gen_query('COUNT(*)', 'stored_listen_data', '"user"', user_id)[0][0])

    def get_user_stored_private_data(self,  user_id):
        '''
        INPUT: user_id
        OUTPUT: DataFrame of stored data for user
        '''
        user_data = self.gen_query('*', 'stored_listen_data_private', '"user"', user_id)
        df = pd.DataFrame(user_data)
        df.columns = ['user', 'artist_id', 'play_count']
        return df

    def count_user_stored_private_data(self, user_id):
        '''
        INPUT: user_id
        OUTPUT: int, count of stored data for user
        '''
        return int(self.gen_query('COUNT(*)', 'stored_listen_data_private', '"user"', user_id)[0][0])


    def get_song_info(self, song_id):
        '''
        INPUT: song id or list of song_ids
        OUTPUT: list of tuples with song name and artist name
        '''

        st = ''
        for song in song_id:
            st += "'%s'," % song
        st = st[:-1]

        self.c.execute(
        '''
        SELECT artist_name, title FROM songs WHERE song_id IN (%s);
        '''
        % st
        )
        self.conn.commit()
        return self.c.fetchall()

    def get_artist_ids(self, list_artist_names):
        ''' should improve later to do regex etc and get full data'''
        if type(list_artist_names) != list:
            print 'input must be a list!'
            raise Exception
        return self.gen_query(select_columns = '*', table = 'artist_name_ids2', filter_column = 'artist_name',
            filter_values = list_artist_names)

    def gen_query(self, select_columns, table, filter_column=None, filter_values=None, limit = None):
        '''
        INPUT: 
        - select_columns: list
        - table: str
        - filter_column: str
        - filter_values: list
        OUTPUT: query results
        '''
        try:
            if (type(select_columns) == str) or (type(select_columns) == np.string_):
                select_columns = [select_columns]
            if (type(filter_values) == str) or (type(filter_values) == np.string_):
                filter_values = [filter_values]


            select_query = ''
            for column in select_columns:
                select_query += column + ", "
            select_query = select_query[:-2]

            where_clause = ''
            if filter_column:
                filter_values_str = ''
                for value in filter_values:
                    filter_values_str += "'%s'," % value

                filter_values_str = filter_values_str[:-1]
                where_clause = 'WHERE %s IN (%s)' % (filter_column, filter_values_str)

            limit_clause = ''
            if limit:
                limit_clause = 'LIMIT %s' % str(limit)

           # print 'SELECT %s FROM %s %s %s;' % (select_query, table, where_clause, limit_clause)

            self.c.execute(
            '''
            SELECT %s FROM %s %s %s;
            '''
            % (select_query, table, where_clause, limit_clause)
            )
            self.conn.commit()
            return self.c.fetchall()
        except Exception, e:
            print e
            self.conn.reset()


        pass
    def get_user_listendata(self, user_id):
        return self.gen_query(select_columns = '*', table = 'triplets_artists', 
            filter_column = '"user"', filter_values = user_id)



    def get_artist_id(self, artist_name):
        try:
            self.c.execute('''
            SELECT artist_id FROM artist_name_ids2 WHERE artist_name = %s LIMIT 1;
            '''
            % ("'" + artist_name + "'")
            )
            self.conn.commit()
            return self.c.fetchall()[0][0]
        except Exception, e:
            print e
            self.conn.reset()

    def get_artist_names(self, artist_ids):
        '''
        INPUT: list of artist ids
        OUTPUT: list of artist names
        '''
        return self.gen_query(select_columns = '*', table = 'artist_name_ids_unique', filter_column = 'artist_id', 
            filter_values = artist_ids)

        pass

    def get_song_artist(self, song_ids):
        '''
        INPUT: list of song_ids
        OUPUT: list of artist_ids, and artist_names
        '''
        return self.gen_query(select_columns = 'artist_id',
            table = 'songs', filter_column = 'song_id', filter_values = song_ids)


    def get_artist_terms(self, artist_id):
        return self.gen_query(select_columns = '*', table = 'artist_term', 
            filter_column = 'artist_id', filter_values = artist_ids)
        pass

    def pivot_user_table(userids, index, columns, values):
        '''
        INPUT: list of userids
        OUTPUT: 
        '''
        pass



    def get_similar_artists(self, artist_name):
        artist_id = self.get_artist_id(artist_name)
        self.c.execute('''
        SELECT 
        ''')
        pass
        



if __name__ == '__main__':
    m = MSD_Queries()
    print m.get_artist_id('Stephen Varcoe')
    m.c.close()