import pandas as pd
import datetime
import numpy as np

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

        # Filter also their embeddings
        loader.embeddings_df = loader.embeddings_df[loader.embeddings_df.movieId.isin(rated_movie_ids)]
        loader.embeddings_df = loader.embeddings_df.reset_index(drop=True)

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

        # Filter also their embeddings
        loader.embeddings_df = loader.embeddings_df[loader.embeddings_df.movieId.isin(loader.movies_df.movieId)]
        loader.embeddings_df = loader.embeddings_df.reset_index(drop=True)

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

# Dummy loader to simulate the object used in filters
class Loader:
    def __init__(self, movies_df, ratings_df, links_df, embeddings_df=None, genres_embeddings=None, plot_embeddings=None):
        movies_df.loc[:, "year"] = movies_df.title.apply(self._parse_year)
        self.movies_df = movies_df
        self.movies_df["genres"] = movies_df["genres"].str.replace("|", ", ", regex=False)
        self.ratings_df = ratings_df
        self.links_df = links_df
        self.embeddings_df = embeddings_df if embeddings_df is not None else pd.DataFrame(columns=["movieId"])
        self.genres_embeddings = genres_embeddings
        self.plot_embeddings = plot_embeddings

    def _parse_year(self, x):
        x = x.split("(")
        if len(x) <= 1:
            return 0
        try:
            return int(x[-1].split(")")[0])
        except:
            return 0

def create_loaders():
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")
    links_df = pd.read_csv("links.csv")
    genre_embeddings = np.load("genres_embeddings.npy")
    plot_embeddings = np.load("plot_embeddings.npy")

    demo_genres_embeddings = np.load("demo_genres_embeddings.npy")
    demo_plot_embeddings = np.load("demo_plot_embeddings.npy")
    

    explore_movies_df = movies_df.copy()
    explore_ratings_df = ratings_df.copy()
    explore_links_df = links_df.copy()

    demo_loader = Loader(movies_df, ratings_df, links_df, genres_embeddings=demo_genres_embeddings, plot_embeddings=demo_plot_embeddings)
    experiment_loader = Loader(explore_movies_df, explore_ratings_df, explore_links_df, genres_embeddings=genre_embeddings, plot_embeddings=plot_embeddings)

    filters_demo = [
        RatingFilterOld(2017),                             
        RatingsPerYearFilter(200),                        
        RatingUserFilter(200),                              
        MoviesNoGenreFilter(),                           
        RatedMovieFilter(),      
        LinkFilter()              
    ]

    filters_experiment = [                          
        RatingFilterOld(2017),                             
        RatingsPerYearFilter(10),                        
        RatingUserFilter(50),                              
        MoviesNoGenreFilter(),                           
        RatedMovieFilter(),                                 
        LinkFilter()                                        
    ]

    for f in filters_demo:
        f(demo_loader)

    for f in filters_experiment:
        f(experiment_loader)

    return demo_loader, experiment_loader