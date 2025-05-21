import datetime
import time
from imdb import Cinemagoer

import pandas as pd
import numpy as np
from collections import defaultdict
import os


from scipy.spatial.distance import squareform, pdist

from PIL import Image
from io import BytesIO
import requests

# from app import pm
from flask import url_for, has_app_context

# Movie-lens data loader

class RatingUserFilter:
    def __init__(self, min_ratings_per_user):
        self.min_ratings_per_user = min_ratings_per_user

    def __call__(self, loader):
        loader.ratings_df = loader.ratings_df[loader.ratings_df['userId'].map(loader.ratings_df['userId'].value_counts()) >= self.min_ratings_per_user]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
    

class RatedMovieFilter:
    def __call__(self, loader):
        # We are only interested in movies for which we have ratings
        rated_movie_ids = loader.ratings_df.movieId.unique()

        loader.movies_df = loader.movies_df[loader.movies_df.movieId.isin(rated_movie_ids)]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

# Filters out all ratings of movies that do not have enough ratings per year
class RatingsPerYearFilter:
    def __init__(self, min_ratings_per_year):
        self.min_ratings_per_year = min_ratings_per_year

    def __call__(self, loader):
        movies_df_indexed = loader.movies_df.set_index("movieId")


        oldest_rating = loader.ratings_df.timestamp.min()
        oldest_year = datetime.datetime.fromtimestamp(oldest_rating).year
        print(oldest_year)

        # Add column with age of each movie
        movies_df_indexed.loc[:, "age"] = movies_df_indexed.year.max() - oldest_year

        
        # Calculate number of ratings per year for each of the movies
        loader.ratings_df.loc[:, "ratings_per_year"] = loader.ratings_df['movieId'].map(loader.ratings_df['movieId'].value_counts()) / loader.ratings_df['movieId'].map(movies_df_indexed["age"])
        
        # Filter out movies that do not have enough yearly ratings
        loader.ratings_df = loader.ratings_df[loader.ratings_df.ratings_per_year >= self.min_ratings_per_year]

class MoviesNoGenreFilter:
    def __call__(self, loader):
        # Filter out movies with no genres
        movie_ids_with_no_genres = loader.movies_df[loader.movies_df.genres == '(no genres listed)'].movieId

        loader.movies_df = loader.movies_df[~loader.movies_df.movieId.isin(movie_ids_with_no_genres)]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

        # Filter out ratings of movies with no genres
        loader.ratings_df = loader.ratings_df[loader.ratings_df.movieId.isin(loader.movies_df.movieId)]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)

        print(f"Ratings shape after filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")


class RatingFilterOld:
    def __init__(self, oldest_rating_year):
        self.oldest_rating_year = oldest_rating_year
    def __call__(self, loader):
        # Marker for oldest rating
        oldest_rating = datetime.datetime(year=self.oldest_rating_year, month=1, day=1, tzinfo=datetime.timezone.utc).timestamp()
        # Filter ratings that are too old
        loader.ratings_df = loader.ratings_df[loader.ratings_df.timestamp > oldest_rating]
        #loader.ratings_df = loader.ratings_df.reset_index(drop=True)

class LinkFilter:
    def __call__(self, loader):
        loader.links_df = loader.links_df[loader.links_df.index.isin((loader.movies_df.movieId))]

class MLDataLoader:
    def __init__(self, ratings_path, movies_path, genres_embeddings_path, plot_embeddings_path, tags_path, links_path,
        filters = None, rating_matrix_path = None, img_dir_path = None):

        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.genres_embeddings_path = genres_embeddings_path
        self.plot_embeddings_path = plot_embeddings_path
        self.tags_path = tags_path
        self.filters = filters
        self.links_path = links_path
        self.rating_matrix_path = rating_matrix_path

        self.ratings_df = None
        self.movies_df = None
        self.movies_df_indexed = None
        self.genres_embeddings = None
        self.plot_embeddings = None
        self.tags_df = None
        self.links_df = None
        self.rating_matrix = None
        self.movie_index_to_id = None
        self.movie_id_to_index = None
        self.num_movies = None
        self.num_users = None
        self.user_to_user_index = None
        self.movie_index_to_description = None
        self.tag_counts_per_movie = None

        self.access = Cinemagoer()
        self.movie_index_to_url = dict()
        self.similarity_matrix = None

        self.img_dir_path = img_dir_path        

    def _get_image(self, imdbId):
        try:
            return self.access.get_movie(imdbId)["full-size cover url"]
        except Exception as e:
            print(f"@@ Exception e={e}")
            return ""

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle imdb
        del state["access"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.access = Cinemagoer()

    # Download all the images
    def download_images(self):
        i = 0
        start_time = time.perf_counter()
        for movie_idx, url in self.movie_index_to_url.items():
            
            if i % 100 == 0:
                print(f"{i}/{len(self.movie_index_to_url)} images were downloaded, took: {time.perf_counter() - start_time}")
                start_time = time.perf_counter()

            movie_id = self.movie_index_to_id[movie_idx]
            
            err = False
            try:
                resp = requests.get(url, stream=True)
            except:
                err = True

            if err or resp.status_code != 200:
                print(f"Error downloading image from url={url} for movie_id={movie_id}")
                # Try once more retrieving fresh url for the given movie
                imdbId = self.links_df.loc[movie_id].imdbId
                new_url = self._get_image(imdbId)

                err = False
                try:
                    resp = requests.get(new_url, stream=True)
                except:
                    err = True

                if err or resp.status_code != 200:
                    print(f"Even new URL={new_url} failed")
                    continue # Ignore the movie
                else:
                    print("New URL succeeded")

            img = Image.open(BytesIO(resp.content))
            width, height = img.size
            TARGET_WIDTH = 200
            coef = TARGET_WIDTH / width
            new_height = int(height * coef)
            img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS).convert('RGB')
            img.save(os.path.join(self.img_dir_path, f'{movie_id}.jpg'), quality=90)

            i += 1

    def get_image(self, movie_idx):
        if self.img_dir_path and has_app_context():
            if movie_idx not in self.movie_index_to_url:
                # Download it first if it is missing
                movie_id = self.movie_index_to_id[movie_idx]
                imdbId = self.links_df.loc[movie_id].imdbId
                remote_url = self._get_image(imdbId)
                
                err = False
                try:
                    resp = requests.get(remote_url, stream=True)
                except:
                    err = True

                if err or resp.status_code != 200:
                    print(f"Failed download image for movie with id={movie_id}")
                else:
                    img = Image.open(BytesIO(resp.content))
                    width, height = img.size
                    TARGET_WIDTH = 200
                    coef = TARGET_WIDTH / width
                    new_height = int(height * coef)
                    img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS).convert('RGB')
                    img.save(os.path.join(self.img_dir_path, f'{movie_id}.jpg'), quality=90)

            # Use local version of images
            item_id = self.movie_index_to_id[movie_idx]
            return url_for('static', filename=f'datasets/ml-latest/img/{item_id}.jpg')

        return self.movie_index_to_url[movie_idx]
        #movie_id = self.movie_index_to_id[movie_idx]
        #return self._get_image(self.links_df.loc[movie_id].imdbId)

    def apply_tag_filter(self, tag_filter, *args, **kwargs):
        self.tags_df = tag_filter(self.tags_df, *args, **kwargs)

        # Set of all unique tags
        self.tags = set(self.tags_df.tag.unique())
        # Maps movie index to tag counts per movie
        self.tag_counts_per_movie = { movie_index : defaultdict(int) for movie_index in self.movie_index_to_id.keys() }
        for group_name, group_df in self.tags_df.groupby("movieId"):
            for _, row in group_df.iterrows():
                self.tag_counts_per_movie[self.movie_id_to_index[group_name]][row.tag] += 1

    def _parse_year(self, x):
        x = x.split("(")
        if len(x) <= 1:
            return 0
        try:
            return int(x[-1].split(")")[0])
        except:
            return 0

    # Passing local_movie_images as parameter to prevent cache changes
    # If local_movie_images==True, we will use local files instead of image urls inside img uris
    def load(self):
        #### Data Loading ####

        # Load ratings
        self.ratings_df = pd.read_csv(self.ratings_path)
        print(f"Ratings shape: {self.ratings_df.shape}, n_users = {self.ratings_df.userId.unique().size}, n_items = {self.ratings_df.movieId.unique().size}")
        
        # Load tags and convert them to lower case
        self.tags_df = pd.read_csv(self.tags_path)
        self.tags_df.tag = self.tags_df.tag.str.casefold()

        # Load movies
        self.movies_df = pd.read_csv(self.movies_path)

        self.movies_df.loc[:, "year"] = self.movies_df.title.apply(self._parse_year)

        # Load embeddings
        self.genres_embeddings = np.load(self.genres_embeddings_path)
        self.plot_embeddings = np.load(self.plot_embeddings_path)

        # Load links
        self.links_df = pd.read_csv(self.links_path, index_col=0)
        
        #### Filtering ####

        # # Filter rating dataframe
        # if self.ratings_df_filter:
        #     self.ratings_df = self.ratings_df_filter(self.ratings_df, self)

        # # Filter movies dataframe
        # if self.movies_df_filter:
        #     self.movies_df = self.movies_df_filter(self.movies_df, self.ratings_df)

        for f in self.filters:
            f(self)

        self.movie_index_to_id = pd.Series(self.movies_df.movieId.values,index=self.movies_df.index).to_dict()
        self.movie_id_to_index = pd.Series(self.movies_df.index,index=self.movies_df.movieId.values).to_dict()
        num_movies = len(self.movie_id_to_index)

        # # Filter tags dataframe
        # if self.tags_df_filter:
        #     self.apply_tag_filter(self.tags_df_filter, self.ratings_df)


        unique_users = self.ratings_df.userId.unique()
        num_users = unique_users.size
        
        self.user_to_user_index = dict(zip(unique_users, range(num_users)))

        # ratings_df_i = self.ratings_df.copy()
        # ratings_df_i.userId = ratings_df_i.userId.map(self.user_to_user_index)
        # ratings_df_i.movieId = ratings_df_i.movieId.map(self.movie_id_to_index)
        # self.rating_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values="rating").fillna(0).values
        # self.similarity_matrix = np.float32(squareform(pdist(self.rating_matrix.T, "cosine")))
        
        # Maps movie index to text description
        self.movies_df["description"] = self.movies_df.title + ' ' + self.movies_df.genres
        self.movie_index_to_description = dict(zip(self.movies_df.index, self.movies_df.description))
        

        self.movies_df_indexed = self.movies_df.set_index("movieId")


        # First check which images are downloaded so far
        already_downloaded = [] if not os.path.exists(self.img_dir_path) else os.listdir(self.img_dir_path)
        self.movie_index_to_url = dict()
        for img_name in already_downloaded:
            movie_id = int(img_name.split(".jpg")[0])
            if movie_id in self.movie_id_to_index: # It could be possible we have more images than movies (due to filtering)
                self.movie_index_to_url[self.movie_id_to_index[movie_id]] = os.path.join(self.img_dir_path, img_name)

        print(f"Already downloaded images for: {len(self.movie_index_to_url)} movies")
        
        # GET IDS OF NEW MOVIES
        # cover_ids = []
        # for index, value in self.movie_index_to_id.items():
        #     if index not in self.movie_index_to_url:
        #         cover_ids.append(value)
        # print(cover_ids)

        return True