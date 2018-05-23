# load to vertica

import utils
import pandas as pd
import dask.dataframe as dd
import os
import pandas
import tempfile

conn, cur = utils.connect_to_db()
# conn, cur = utils.connect_to_db()
os.chdir('/home/jo/Documents/Master thesis @ BUX/notebooks/scoring')
predictions = dd.read_csv('*.csv').compute()
print("Number of users:", len(predictions))

def copy_to_vertica(source_df, destination_table, connection, include_index=False, truncate=False, verbose=False):

    if truncate == True:
        print("Truncating table " + destination_table)
        cur = connection.cursor()
        cur.execute("truncate table " + destination_table)

    columns = list(source_df)
    
    if include_index == True:
        columns.insert(0, source_df.index.name)
    
    columns = " " + str(tuple(source_df)).replace("'","")
        
    tmp_file = tempfile.NamedTemporaryFile()
    source_df.to_csv(tmp_file.name, sep = ',', index = include_index)
    cur = connection.cursor()
    sql = "copy " + destination_table + columns + " from stdin delimiter ',' skip 1"
    
    if verbose == True:
        print(sql)
    
    with open(tmp_file.name, "rb") as fs:
        cur.copy(sql, fs)
        connection.commit()
    tmp_file.close()
    
    return str(len(source_df)) + ' row(s) written to table ' + destination_table


copy_to_vertica(predictions, 'analytics.model_scoring_predictions', conn)