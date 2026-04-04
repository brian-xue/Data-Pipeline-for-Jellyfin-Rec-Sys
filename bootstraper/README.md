# <center> Bootstrap Dataset For the Jellyfin Recommendation System </center>


#### This docker will only be executed during the initial stage of the system
## Docker Orchestra
The structure of the docker is below 
```
bootstraper/
    scripts/
        build_embedding_index.py
        build_embedding_text.py
        build_initial_user.py
        config.yaml
        embedding.py
        ingest_datasets.py
        minio_s3.py
    .dockerignore
    Dockerfile
    requirements.txt
    run_all.sh
```
## WorkFlow
### Ingest Dateset
This function will be performed by the file `ingest_datasets.py`. It will extract the 2 external datasets and load them into the object store.
The datasets are:
    - movieLen32M
    - TMDBdataset
To get the TMDB dataset, you will need to sign up for a free API key at Kaggle, and export it in the enviroment `export KAGGLE_API_TOKEN=your-api-key`. After downloaded, the program will check the checksum for the MovieLens32M. Then upload both dataset into s3 storage.

You should see the following files in s3 storage after executing the program:
`links.csv`, `movies.csv`,  `ratings.csv`, `tags.csv`, `TMDB_movie_dataset_v11.csv`.

### Initial Process for the Data
To fully extract the features from the ingested datasets, we want to use a model to extract embeddings. But first, we need to do a initial process to prepare the data for embedding extraction. This will be performed by `build_embedding_text.py`. In this program, it will first extract the `links.csv`, `movies.csv`, `tags.csv` and `TMDB_movie_dataset_v11.csv` files from the object store. Then, the program will combine them together, and generate a new schema with following columns ` "movieId","imdbId","tmdbId", "title","original_title","release_year","adult",       "original_language","genres_list","keywords_list","top_user_tags",       "spoken_languages_list","production_countries_list",       "production_companies_list", "tagline","overview","embedding_text"`. We believe this will help to find the key factors that people want to watch some movies.

This program will store the output in both `jsonl` and `parquet` format, for inspect and sufficient use.

### Embedding
In this part, we will use the model to generate the semantic embeddings by `embedding.py`. The model is `sentence-transformers/all-MiniLM-L6-v2`. It will build weighed embeddings, and the genres and overview account for the majority of the semantic information. After embedding, the result will be uploaded to s3.

### Embedding Indexing
In this part, we will create an index for the generated embeddings to enable efficient retrieval during recommendation by `build_embedding_index.py`. I use the `faiss` library to build the index. So when we need the candidates during inference, this can help to retrieve the candidates efficiently. Also, the indexing will be saved at block storage, so it can be loaded much faster than s3.

### Build User profiles
This step is performed by `build_initial_user.py`.

It reads the cleaned ratings and movie embeddings, then builds four bootstrap artifacts:
- `base_user_events.parquet`: early user interactions used as profile history.
- `remaining_user_events.parquet`: later interactions kept for training/evaluation.
- `base_users.parquet`: user-level statistics (interaction counts, rating stats, activity span).
- `base_user_profiles.parquet`: long-term and short-term user embeddings.

Main logic:
- Split each user timeline into bootstrap and remaining parts.
- Compute time-weighted preference signals from ratings.
- Aggregate movie embeddings into user profile vectors.
- Write parquet outputs and upload to object storage.

This output becomes the initial user state for downstream training and online feature building.


## Running Scripts
#### Inside the docker
```sh
python scripts/ingest_datasets.py
python scripts/build_embedding_text.py
python scripts/embedding.py
python scripts/build_embedding_index.py
python scripts/build_initial_user.py --config scripts/config.yaml
```
#### Outside the docker
```sh
docker exec bootstraper bash /app/run_all.sh
```
