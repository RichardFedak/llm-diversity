import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

movies = [

    {"title": "The Dark Knight", "genres": ["Action", "Crime", "Drama"],
     "plot": "Batman faces off against the Joker, a criminal mastermind who wants to plunge Gotham into chaos."},

    {"title": "Forrest Gump", "genres": ["Drama", "Romance"],
     "plot": "The life journey of a kind-hearted man intersects with major historical events through decades of U.S. history."},

    {"title": "The Matrix", "genres": ["Action", "Sci-Fi"],
     "plot": "A computer hacker discovers reality is a simulation and joins a rebellion against its controllers."},

    {"title": "Inception", "genres": ["Action", "Adventure", "Sci-Fi"],
     "plot": "A thief who infiltrates dreams is hired for a risky mission to plant an idea into a person's subconscious."},

    {"title": "Finding Nemo", "genres": ["Animation", "Adventure", "Comedy", "Family"],
     "plot": "A timid clownfish travels across the ocean to rescue his captured son."},

    {"title": "The Lion King", "genres": ["Animation", "Adventure", "Drama", "Family"],
     "plot": "A young lion prince flees his kingdom after tragedy and learns to embrace his destiny."},

    {"title": "Titanic", "genres": ["Drama", "Romance"],
     "plot": "A young couple from different social classes fall in love aboard the doomed RMS Titanic."},

    {"title": "The Silence of the Lambs", "genres": ["Crime", "Drama", "Thriller"],
     "plot": "An FBI trainee consults a brilliant but imprisoned killer to catch another serial murderer."},
     
]


test_movie_genres = {
    "title": "Interstellar",
    "genres": ["Adventure", "Drama", "Sci-Fi"],
    "plot": "A team of explorers travels through a wormhole in space to save humanity from environmental collapse."
}


test_movie_plot = {
    "title": "Taken",
    "genres": ["Action", "Thriller"],
    "plot": "A former spy uses his skills to track down and rescue his daughter after she's abducted by traffickers."
}



def cluster_and_visualize(genre_weight=1.0, plot_weight=1.0, k=2):
    genre_emb = np.array(model.encode([ "Genres of the movie: "+ ", ".join(movie['genres']) for movie in movies])) * genre_weight
    plot_emb = np.array(model.encode(["Plot of the movie: " + movie['plot'] for movie in movies])) * plot_weight

    embeddings = (genre_emb + plot_emb) / 2

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    centroids_reduced = pca.transform(kmeans.cluster_centers_)


    def embed_movie(movie):
        genre_emb = model.encode([', '.join(movie['genres'])]) * genre_weight
        plot_emb = model.encode([movie['plot']]) * plot_weight
        return genre_emb + plot_emb

    test_emb = embed_movie(test_movie_genres)
    test_similarities = cosine_similarity(test_emb, kmeans.cluster_centers_)
    test_cluster = np.argmax(test_similarities)

    test_movie_reduced = pca.transform(test_emb)
    plt.scatter(test_movie_reduced[:, 0], test_movie_reduced[:, 1], c=cluster_labels[test_cluster], marker='o', s=100, label=f"Genres (Cluster {test_cluster})")
    plt.text(test_movie_reduced[0, 0] + 0.03, test_movie_reduced[0, 1] + 0.03, test_movie_genres['title'], fontsize=7)

    test_emb = embed_movie(test_movie_plot)
    test_similarities = cosine_similarity(test_emb, kmeans.cluster_centers_)
    test_cluster = np.argmax(test_similarities)

    test_movie_reduced = pca.transform(test_emb)
    plt.scatter(test_movie_reduced[:, 0], test_movie_reduced[:, 1], c=cluster_labels[test_cluster], marker='o', s=100, label=f"Plot (Cluster {test_cluster})")
    plt.text(test_movie_reduced[0, 0] + 0.03, test_movie_reduced[0, 1] + 0.03, test_movie_plot['title'], fontsize=7)

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='x', s=200, c='red', label='Centroids')


    for i, movie in enumerate(movies):
        plt.text(reduced_embeddings[i, 0] + 0.03, reduced_embeddings[i, 1] + 0.03, movie['title'], fontsize=7)

    plt.title(f"Genre Weight: {genre_weight}, Plot Weight: {plot_weight}")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

plt.figure(figsize=(14, 4))

weight_settings = [
    (1.0, 1.0),
    (0.5, 2),
    (2, 0.5)
]

for i, (gw, pw) in enumerate(weight_settings):
    plt.subplot(1, 3, i + 1)
    cluster_and_visualize(genre_weight=gw, plot_weight=pw)

plt.tight_layout()
plt.show()
