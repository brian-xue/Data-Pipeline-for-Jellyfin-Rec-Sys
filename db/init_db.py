import psycopg2

conn = psycopg2.connect(
    host="localhost",      
    port=5432,
    user="recsys",
    password="ecsys123",  
    dbname="recsys"        
)

with conn:
    with conn.cursor() as cur:
        with open("db/init/create_tables.sql", "r") as f:
            sql = f.read()
            cur.execute(sql)

print("✅ Tables initialized successfully.")