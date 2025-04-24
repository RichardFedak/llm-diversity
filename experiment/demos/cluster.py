import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2') 

movies = [
    {"title": "Iron Man", "genres": ["Action", "Adventure", "Sci-Fi"], 
        "plot": "A wealthy industrialist is kidnapped and builds a high-tech suit of armor to escape."},
    
    {"title": "Captain America: The First Avenger", "genres": ["Action", "Adventure", "Sci-Fi"], 
        "plot": "A scrawny but determined man becomes a super soldier during World War II."},
    
    {"title": "Thor", "genres": ["Action", "Adventure", "Fantasy"], 
        "plot": "A powerful god is banished to Earth and must learn humility to reclaim his throne."},
    
    {"title": "Dunkirk", "genres": ["Action", "Drama", "War"], 
        "plot": "Allied soldiers from Belgium, the British Empire, and France are surrounded by the German army and evacuated during a fierce battle in World War II."},
    
    {"title": "Mad Max: Fury Road", "genres": ["Action", "Adventure", "Sci-Fi"], 
        "plot": "In a post-apocalyptic wasteland, a man teams up with a woman to escape a tyrannical warlord."},
    
    {"title": "The Dark Knight", "genres": ["Action", "Crime", "Drama"], 
        "plot": "Batman faces off against the Joker, a criminal mastermind who wants to create chaos in Gotham City."},

    {"title": "Saving Private Ryan", "genres": ["Drama", "War"], 
        "plot": "A group of soldiers is sent to find and bring home a paratrooper whose brothers have been killed in action."},
    
    {"title": "The Shawshank Redemption", "genres": ["Drama"], 
        "plot": "Two imprisoned men bond over the years, finding solace and eventual redemption through acts of common decency."},
    
    {"title": "Forrest Gump", "genres": ["Drama", "Romance"], 
        "plot": "The presidencies of Kennedy and Johnson, the events of Vietnam, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an extraordinary life."},
    
    {"title": "A Beautiful Mind", "genres": ["Drama", "Biography"], 
        "plot": "A brilliant but asocial mathematician struggles with mental illness while developing a revolutionary theory."},
    
    {"title": "The Pursuit of Happyness", "genres": ["Drama", "Biography"], 
        "plot": "A struggling salesman takes custody of his son as he's poised to begin a life-changing professional career."},
    
    {"title": "The Hangover", "genres": ["Comedy", "Adventure"], 
        "plot": "Three friends wake up from a bachelor party in Las Vegas, with no memory of the previous night and the groom missing."},
    
    {"title": "Superbad", "genres": ["Comedy", "Teen"], 
        "plot": "Two high school friends try to enjoy their final days before graduation while dealing with their own issues of friendship and love."},
    
    {"title": "The Grand Budapest Hotel", "genres": ["Comedy", "Drama", "Crime"], 
        "plot": "A hotel concierge and his protégé become involved in a series of misadventures involving a stolen painting and a family fortune."},
    
    {"title": "Dumb and Dumber", "genres": ["Comedy", "Adventure"], 
        "plot": "The cross-country adventures of two well-meaning but dimwitted friends who try to return a lost suitcase to its rightful owner."},
    
    {"title": "Ferris Bueller's Day Off", "genres": ["Comedy", "Teen"], 
        "plot": "A high school student skips school for a day of fun in Chicago, all while avoiding his principal and his sister."},
    
    {"title": "Anchorman: The Legend of Ron Burgundy", "genres": ["Comedy"], 
        "plot": "The absurd antics of an anchorman in a 1970s San Diego newsroom and his battle for dominance in the competitive world of broadcast journalism."}        
]

title_embedding = model.encode([movie['title'] for movie in movies])
genre_embedding = model.encode([', '.join(movie['genres']) for movie in movies])

weighted_genre_embeddings = [genre * 2 for genre in genre_embedding] 

embeddings = title_embedding + weighted_genre_embeddings

inertia = []
for k in range(1, 12):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

print("Inertia values for K = 1 to 5:", inertia)

k = 3 

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_

print("\nCluster assignments for each movie:")
for i, movie in enumerate(movies):
    print(f"{movie['title']} - Cluster {cluster_labels[i]}")

close_movie = {
    "title": "Avengers: Endgame",
    "genres": ["Action", "Adventure", "Sci-Fi"],
    "plot": "The Avengers work to undo the damage caused by Thanos."
}

far_movie = {
    "title": "Finding Nemo",
    "genres": ["Animation", "Adventure", "Comedy", "Family"],
    "plot": "A clownfish embarks on a dangerous journey to find his son, who was captured by a diver and placed in a fish tank."
}

title_embedding = model.encode([close_movie['title']])
genre_embedding = model.encode([', '.join(close_movie['genres'])])
plot_embedding = model.encode([close_movie['plot']])

weighted_genre_embeddings = [genre * 2 for genre in genre_embedding] 

close_embedding = title_embedding + weighted_genre_embeddings + plot_embedding

title_embedding = model.encode([far_movie['title']])
genre_embedding = model.encode([', '.join(far_movie['genres'])])
plot_embedding = model.encode([far_movie['plot']])

weighted_genre_embeddings = [genre * 2 for genre in genre_embedding] 

far_embedding = title_embedding + weighted_genre_embeddings + plot_embedding

cluster_centroids = kmeans.cluster_centers_

close_similarities = cosine_similarity(close_embedding, cluster_centroids)
far_similarities = cosine_similarity(far_embedding, cluster_centroids)

print("\nCosine similarity of 'Avengers: Endgame' (close movie) to each cluster centroid:")
print(close_similarities)

print("\nCosine similarity of 'Finding Nemo' (far movie) to each cluster centroid:")
print(far_similarities)

close_cluster = np.argmax(close_similarities)

far_cluster = np.argmax(far_similarities)

print(f"\n'Avengers: Endgame' is closest to Cluster {close_cluster}")
print(f"'Finding Nemo' is closest to Cluster {far_cluster}")

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=50)

centroids_reduced = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_reduced[:, 0], centroids_reduced[:, 1], marker='x', s=200, c='red', label='Centroids')

close_movie_reduced = pca.transform(close_embedding)
far_movie_reduced = pca.transform(far_embedding)

plt.scatter(close_movie_reduced[:, 0], close_movie_reduced[:, 1], c='blue', marker='o', s=100, label='Close Movie (Avengers: Endgame)')
plt.scatter(far_movie_reduced[:, 0], far_movie_reduced[:, 1], c='orange', marker='o', s=100, label='Far Movie (Finding Nemo)')

plt.title('2D Visualization of Movie Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.legend()

plt.show()