import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

movies = [

    {"title": "Iron Man", "genres": ["Action", "Adventure", "Sci-Fi"],
        "plot": "A wealthy industrialist is kidnapped and builds a high-tech suit to escape."},
    {"title": "Edge of Tomorrow", "genres": ["Action", "Adventure", "Sci-Fi"],
        "plot": "A soldier caught in a time loop relives the same brutal battle against aliens."},
    {"title": "Pacific Rim", "genres": ["Action", "Adventure", "Sci-Fi"],
        "plot": "Giant robots piloted by humans fight massive sea monsters to save the world."},
    {"title": "Ready Player One", "genres": ["Action", "Adventure", "Sci-Fi"],
        "plot": "A teenager enters a virtual reality world to compete in a game that will decide the future of the OASIS."},
    


    {"title": "Saving Private Ryan", "genres": ["Drama", "War"],
        "plot": "A group of soldiers is sent to rescue a paratrooper behind enemy lines during WWII."},
    {"title": "Finding Nemo", "genres": ["Animation", "Adventure", "Comedy", "Family"],
        "plot": "A clownfish crosses the ocean to find his kidnapped son."},
    {"title": "Taken", "genres": ["Action", "Thriller"],
        "plot": "A retired CIA agent uses his skills to rescue his daughter from kidnappers."},
    {"title": "The Searchers", "genres": ["Western", "Adventure", "Drama"],
        "plot": "A Civil War veteran spends years searching for his abducted niece."},
    
]

test_movie_genres = {"title": "The Fifth Element", "genres": ["Action", "Adventure", "Sci-Fi"],
        "plot": "In a futuristic world, a cab driver helps a mystical being save Earth from destruction."}


test_movie_plot = {"title": "Room", "genres": ["Drama", "Thriller"],
        "plot": "A mother and her young son plan their escape from captivity after years of imprisonment."}



def cluster_and_visualize(genre_weight=1.0, plot_weight=1.0, k=2):
    title_emb = model.encode([movie['title'] for movie in movies])
    genre_emb = model.encode([', '.join(movie['genres']) for movie in movies])
    plot_emb = model.encode([movie['plot'] for movie in movies])

    weighted_genre_emb = [g * genre_weight for g in genre_emb]
    weighted_plot_emb = [p * plot_weight for p in plot_emb]

    embeddings = title_emb + weighted_genre_emb + weighted_plot_emb

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    centroids_reduced = pca.transform(kmeans.cluster_centers_)


    def embed_movie(movie):
        title_emb = model.encode([movie['title']])
        genre_emb = model.encode([', '.join(movie['genres'])]) * genre_weight
        plot_emb = model.encode([movie['plot']]) * plot_weight
        return title_emb + genre_emb + plot_emb

    test_emb = embed_movie(test_movie_genres)
    test_similarities = cosine_similarity(test_emb, kmeans.cluster_centers_)
    test_cluster = np.argmax(test_similarities)

    test_movie_reduced = pca.transform(test_emb)
    plt.scatter(test_movie_reduced[:, 0], test_movie_reduced[:, 1], c='blue', marker='o', s=100, label=f"Genres (Cluster {test_cluster})")

    test_emb = embed_movie(test_movie_plot)
    test_similarities = cosine_similarity(test_emb, kmeans.cluster_centers_)
    test_cluster = np.argmax(test_similarities)

    test_movie_reduced = pca.transform(test_emb)
    plt.scatter(test_movie_reduced[:, 0], test_movie_reduced[:, 1], c='green', marker='o', s=100, label=f"Plot (Cluster {test_cluster})")

    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='x', s=200, c='red', label='Centroids')


    for i, movie in enumerate(movies):
        plt.text(reduced_embeddings[i, 0] + 0.1, reduced_embeddings[i, 1] + 0.1, movie['title'], fontsize=7)

    plt.title(f"Genre Weight: {genre_weight}, Plot Weight: {plot_weight}")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(fontsize='small')

plt.figure(figsize=(13, 5))

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
