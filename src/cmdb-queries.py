from dotenv import load_dotenv
from pathlib import Path
import psycopg2
import os
import pandas as pd

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

df=get_data_cmdb("""
        WITH q AS (
            SELECT 'Ramu Mehta' AS given_name
        )
        SELECT
            u.user_name,o.user_id,u.user_phone,o.processing_at,o.order_status
        FROM orders o join tbl_user u on u.user_id=o.user_id, q
        WHERE soundex(u.user_name) = soundex(given_name)
        AND levenshtein(lower(u.user_name),lower(given_name)) <= 2 and o.created_at > now() - interval '30 days';
        """)

# df=get_data_cmdb("""
#         WITH q AS (
#             SELECT 'Ramu Mehta' AS given_name
#         )
#         SELECT
#             tl.name,o.user_id, tl.phone_number, o.processing_at,o.order_status
#         FROM orders o join team_leaders tl on tl.id = o.team_leader, q
#         WHERE soundex(tl.name) = soundex(given_name)
#         AND levenshtein(lower(tl.name),lower(given_name)) <= 2 and o.created_at > now() - interval '30 days';
#         """)

print (df)
