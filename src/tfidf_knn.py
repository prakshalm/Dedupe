import os
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
from pathlib import Path
# Import module for Fuzzy string matching
from fuzzywuzzy import fuzz, process
# Import module for regex
import re
# Import module for iteration
import itertools
from typing import Tuple
# Import module for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Import module for cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
# Import module for KNN
from sklearn.neighbors import NearestNeighbors



env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

def get_data_cmdb(query):

    conn = psycopg2.connect(host=os.environ['HOST'], database=os.environ['DATABASE'], port=os.environ['PORT'],
                      user=os.environ['CMDB_USERNAME'], password=os.environ['CMDB_PASSWORD'])
    print('Connected to Replica DB')
    df = pd.read_sql_query(query, con=conn)
    print('Number of rows in Data - ' + str(df.shape[0]))
    conn.close()
    return df

# String pre-processing
def preprocess_string(s):
    # Remove spaces between strings with one or two letters
    s = re.sub(r'(?<=\b\w)\s*[ &]\s*(?=\w\b)', '', s)
    return s

# String matching - TF-IDF
def build_vectorizer(
    clean: pd.Series,
    analyzer: str = 'char', 
    ngram_range: Tuple[int, int] = (1, 4), 
    n_neighbors: int = 2, 
    **kwargs
    ) -> Tuple:
    # Create vectorizer
    vectorizer = TfidfVectorizer(analyzer = analyzer, ngram_range = ngram_range, **kwargs)
    X = vectorizer.fit_transform(clean.values.astype('U'))

    # Fit nearest neighbors corpus
    nbrs = NearestNeighbors(n_neighbors = n_neighbors, metric = 'cosine').fit(X)
    return vectorizer, nbrs

# String matching - KNN
def tfidf_nn(
    messy, 
    clean,
    user_id_master:pd.Series, 
    n_neighbors = 2, 
    **kwargs
    ):
    # Fit clean data and transform messy data
    vectorizer, nbrs = build_vectorizer(clean, n_neighbors = n_neighbors, **kwargs)
    input_vec = vectorizer.transform(messy)
    # output_vec=vectorizer.transform(clean)
    # Determine best possible matches
    distances, indices = nbrs.kneighbors(input_vec, n_neighbors = n_neighbors)
    nearest_values = np.array(clean)[indices]
    user_id=np.array(user_id_master)[indices]
    
    return nearest_values, distances,user_id

# String matching - match fuzzy
def find_matches_fuzzy(
    row, 
    match_candidates, 
    limit = 5
    ):
    row_matches = process.extract(
        row, dict(enumerate(match_candidates)), 
        scorer = fuzz.token_sort_ratio, 
        limit = limit
        )
    result = [(row, match[0], match[1]) for match in row_matches]
    return result

# String matching - TF-IDF
def fuzzy_nn_match(
    messy,
    clean,
    column,
    user_id_master:pd.Series,
    col,
    n_neighbors = 100,
    limit = 5, **kwargs):
    
    nearest_values, _ ,user_id_parent= tfidf_nn(messy, clean,user_id_master,n_neighbors, **kwargs)

    results = [find_matches_fuzzy(row, nearest_values[i], limit) for i, row in enumerate(messy)]
    df = pd.DataFrame(itertools.chain.from_iterable(results),
        columns = [column, col, 'Ratio']
        )
    df.insert(1, column='User Id Parent', value=user_id_parent.tolist())
    return df

# String matching - Fuzzy
def fuzzy_tf_idf(
    df: pd.DataFrame,
    column: str,
    clean: pd.Series,
    user_id_master:pd.Series,
    col: str,
    analyzer: str = 'char',
    ngram_range: Tuple[int, int] = (1, 3)
    ) -> pd.Series:
    # Create vectorizer
    messy = df[column].dropna().reset_index(drop = True).astype(str)
    # messy = messy_prep.apply(preprocess_string)
    result = fuzzy_nn_match(messy = messy, clean = clean, column = column,user_id_master=user_id_master, col = col, n_neighbors = 1)
    # Map value from messy to clean
    return result


def res_latLong_auto(
    query_data:pd.DataFrame, #query_data
    user_info:pd.DataFrame #master_data
    ):
    
    data_latlong=query_data[query_data['msite_address_info']=="""{'enable_location': True, 'is_reverse_code': False, 'manually_pick_lat_lng': False}"""]
    data_latlong.reset_index(drop=True,inplace=True)

    data_address=query_data[query_data['msite_address_info']=="""{'enable_location': True, 'is_reverse_code': False, 'manually_pick_lat_lng': False}"""]
    data_address.reset_index(drop=True,inplace=True)

    df_resLatLong= (data_latlong.pipe(fuzzy_tf_idf, # Function and messy data
                        column = 'cx_cordinates', # Messy column in data
                        clean = user_info['cx_cordinates'],# Master data (list)
                        user_id_master=user_info['user_id'], # user_ids Master data
                        col = 'Result_latlong') # Can be customized
                )
    df_resLatLong=pd.concat([ data_latlong['user_id'],df_resLatLong], axis=1)
    # print(df_resLatLong)

    df_resAddress = (data_latlong.pipe(fuzzy_tf_idf, # Function and messy data
                        column = 'cx_formatted_address', # Messy column in data
                        clean = user_info['cx_formatted_address'], # Master data (list)
                        user_id_master=user_info['user_id'], # user_ids Master data
                        col = 'Result_address') # Can be customized
                )
    df_resAddress=pd.concat([ data_latlong['user_id'],df_resAddress], axis=1)
    print(df_resAddress)

    res= pd.merge(df_resLatLong, df_resAddress, how ='inner', on =['user_id'])
    res=res[res['Ratio_x'] >= 80].reset_index(drop=True)
    print(res)    
    res.to_csv('./result_latLong_auto.csv')
    
    return res
    
    
def res_address(
    query_data:pd.DataFrame, #query_data
    user_info:pd.DataFrame #master_data
    ):
    
    df_resLatLong= (query_data.pipe(fuzzy_tf_idf, # Function and messy data
                        column = 'cx_cordinates', # Messy column in data
                        clean = user_info['cx_cordinates'],# Master data (list)
                        user_id_master=user_info['user_id'], # user_ids Master data
                        col = 'Result_latlong') # Can be customized
                )
    df_resLatLong=pd.concat([ query_data['user_id'],df_resLatLong], axis=1)
    # print(df_resLatLong)

    df_resAddress = (query_data.pipe(fuzzy_tf_idf, # Function and messy data
                        column = 'cx_formatted_address', # Messy column in data
                        clean = user_info['cx_formatted_address'], # Master data (list)
                        user_id_master=user_info['user_id'], # user_ids Master data
                        col = 'Result_address') # Can be customized
                )
    df_resAddress=pd.concat([ query_data['user_id'],df_resAddress], axis=1)
    print(df_resAddress)

    res= pd.merge(df_resLatLong, df_resAddress, how ='inner', on =['user_id'])
    res=res[res['Ratio_y'] >= 80].reset_index(drop=True)
    print(res)    
    res.to_csv('./result_latLong_auto.csv')
    
    return res


if __name__=="__main__":
    
    #For location enabled latlong    
    query_data_auto=get_data_cmdb("""Select  concat_ws(', ',o.cx_lat, o.cx_lng) as cx_cordinates,o.user_id,t.msite_address_info ,o.cx_formatted_address,o.created_at from orders o join user_addresses t on o.user_id=t.user_id where o.created_by_type = 'msite' and o.created_at>=date_trunc('day', now() - interval '1 days') and t.msite_address_info is not null """)
    query_data_auto=query_data_auto[query_data_auto['cx_cordinates'].notna()]
    query_data_auto.to_csv('./query.csv')
    query_data_auto=pd.read_csv('./query.csv',index_col=0)


    #For all cases of latlong
    query_data=get_data_cmdb("""Select  concat_ws(', ',o.cx_lat, o.cx_lng) as cx_cordinates,o.user_id,t.msite_address_info ,o.cx_formatted_address,o.created_at from orders o join user_addresses t on o.user_id=t.user_id where o.created_by_type = 'msite' and o.created_at>=date_trunc('day', now() - interval '1 days') and t.msite_address_info is not null """)
    query_data=query_data[query_data['cx_cordinates'].notna()]
    query_data.to_csv('./query.csv')
    query_data=pd.read_csv('./query.csv',index_col=0)

    
    
    user_info=get_data_cmdb("""Select  concat_ws(', ',cx_lat, cx_lng) as cx_cordinates, cx_formatted_address,user_id,created_at from orders where created_by_type = 'msite' and created_at<date_trunc('day', now() - interval '1 days')  and created_at>(date_trunc('day', now() - interval '1 days')- interval '15 days') """ )
    user_info = user_info[user_info['cx_cordinates'].notna()]
    user_info.to_csv('./user_info.csv')
    user_info=pd.read_csv('./master.csv',index_col=0)

    #For enabled_location=True
    res_latLong_auto(query_data=query_data_auto,user_info=user_info)

    res_address(query_data=query_data,user_info=user_info)


