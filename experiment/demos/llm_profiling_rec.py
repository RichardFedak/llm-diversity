import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sentence_transformers import SentenceTransformer
from ollama import chat
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from pydantic import BaseModel

from pydantic import BaseModel
import matplotlib.pyplot as plt
import umap

model = SentenceTransformer('all-MiniLM-L6-v2')

movies = [
    {"title": "Puss in Boots: The Last Wish", "genres": ["Animation", "Family", "Adventure"],
     "plot": "Puss in Boots sets out on an epic journey to find the mythical Last Wish and restore his nine lives."},
    {"title": "Interstellar", "genres": ["Action", "Sci-Fi", "Adventure"],
     "plot": "A group of astronauts travel through a wormhole in search of a new home for humanity."},
    {"title": "Prisoners", "genres": ["Drama", "Mystery", "Thriller"],
     "plot": "When his daughter goes missing, a desperate father takes matters into his own hands as the investigation unfolds."},
    {"title": "Harry Potter and the Sorcerer's Stone", "genres": ["Fantasy", "Adventure", "Family"],
     "plot": "A young boy discovers he is a wizard and begins his first year at Hogwarts School of Witchcraft and Wizardry."},
    {"title": "The Conjuring", "genres": ["Horror", "Thriller", "Mystery"],
     "plot": "Paranormal investigators help a family terrorized by a dark presence in their farmhouse."},
    {"title": "The Notebook", "genres": ["Romance", "Drama"],
     "plot": "A young couple falls in love during the early years of World War II, facing obstacles from society and memory loss."},
]


user_preferred_movies = [
    {"title": "Toy Story 4", "genres": ["Animation", "Family", "Adventure"], "plot": "Woody and Buzz embark on a journey to find a new friend."},
    {"title": "Frozen II", "genres": ["Animation", "Family", "Fantasy"], "plot": "Elsa and Anna travel to an ancient forest to uncover secrets about their past."},
    {"title": "The Lion King", "genres": ["Animation", "Family", "Adventure"], "plot": "A young lion prince must reclaim his throne after his father's untimely death."},
    {"title": "Guardians of the Galaxy", "genres": ["Action", "Sci-Fi", "Adventure"], "plot": "A group of intergalactic misfits team up to save the universe."},
    {"title": "The Matrix", "genres": ["Action", "Sci-Fi", "Thriller"], "plot": "A computer hacker discovers that reality is an illusion controlled by machines."},
    {"title": "Star Wars: A New Hope", "genres": ["Action", "Sci-Fi", "Adventure"], "plot": "A young farm boy joins a rebellion to defeat the evil Empire and restore peace to the galaxy."},
    {"title": "The Pursuit of Happyness", "genres": ["Drama", "Biography"], "plot": "A determined father battles homelessness to build a better future for his son."},
    {"title": "The Imitation Game", "genres": ["Biography", "Drama", "War"], "plot": "A brilliant mathematician races to crack Nazi codes during World War II while hiding a personal secret."},
    {"title": "A Beautiful Mind", "genres": ["Drama", "Biography"], "plot": "A gifted mathematician fights mental illness while reshaping economic theory."}
]

class Representant(BaseModel):
    genres: str
    plot: str

user_genre_embeddings = model.encode(["Genres: " + ", ".join(movie['genres']) for movie in user_preferred_movies])
user_plot_embeddings = model.encode(["Plot: " + movie['plot'] for movie in user_preferred_movies])
user_embeddings = (2 * user_genre_embeddings + 0.5 * user_plot_embeddings) / 2

candidate_genre_embeddings = model.encode(["Genres: " + ", ".join(movie['genres']) for movie in movies])
candidate_plot_embeddings = model.encode(["Plot: " + movie['plot'] for movie in movies])
candidate_embeddings = (2 * candidate_genre_embeddings + 0.5 * candidate_plot_embeddings) / 2


print("Running HDBSCAN clustering...")
hdbscan_clusterer = HDBSCAN(
    min_cluster_size=3,
    min_samples=None,
    metric='euclidean',
)
cluster_labels = hdbscan_clusterer.fit_predict(user_embeddings)

unique_labels = set(cluster_labels)
n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise_points = np.sum(np.array(cluster_labels) == -1)

print(f"\nHDBSCAN found {n_clusters_found} cluster(s).")
print(f"Total points: {len(user_embeddings)}")
print(f"Noise points (label -1): {n_noise_points}")
print(f"Cluster labels assigned: {cluster_labels}")

clusters = {}
noise_movies = []
for i in range(len(user_preferred_movies)):
    label = cluster_labels[i]
    movie_info = user_preferred_movies[i]
    if label == -1:
        noise_movies.append({**movie_info, "original_index": i})
    else:
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({**movie_info, "original_index": i})

if n_clusters_found > 0:
    print("\n--- Cluster Movie Lists ---")
    for cluster_id in sorted(clusters.keys()):
        print(f"\nCluster {cluster_id} ({len(clusters[cluster_id])} users/movies):")
        for movie_data in clusters[cluster_id]:
            print(f"- {movie_data.get('title', 'N/A')}")
if n_noise_points > 0:
    print(f"\n--- Noise Points ({n_noise_points} users/movies) ---")
    for movie_data in noise_movies:
         print(f"- {movie_data.get('title', 'N/A')}")

def generate_representant(movies_cluster):
    movies = "\n".join(f"- {movie['title']} | Genres: " + ", ".join(movie['genres']) + "Plot: " + movie["plot"] for movie in movies_cluster)
    
    llama_prompt = f"""
    The user enjoys the following movies:
    
    {movies}
    
    Based on these, write a synthetic movie-style entry that represents the user's movie taste.
    Return a short JSON with 'genres' and 'plot'. Genres should be comma-separated keywords. The plot should match the tone and structure of actual movie plots in the dataset.
    """

    response = chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates user preference profiles for movie recommendation systems. You always return output in a format that matches movie metadata: genres and plot."},
            {"role": "user", "content": llama_prompt}
        ],
        model="mistral",
        format=Representant.model_json_schema(),
    )
    representant = Representant.model_validate_json(response.message.content)

    return representant

representants = {}
representant_embeddings_dict = {}
representant_embeddings_list = []
representant_cluster_ids_list = []

print("\n--- Generating Cluster Representants ---")
for cluster_id, cluster_movies_data in clusters.items():
    print(f"Generating representant for Cluster {cluster_id}...")

    representant = generate_representant([m for m in cluster_movies_data])
    if representant:
        representants[cluster_id] = representant
        print(f"Representant {cluster_id}:", representant)
    
        rep_text = f"Genres: {representant.genres} Plot: {representant.plot}"
        emb = model.encode([rep_text])
        representant_embeddings_dict[cluster_id] = emb[0]
        representant_embeddings_list.append(emb[0])
        representant_cluster_ids_list.append(cluster_id)
    else:
        print(f"Skipping representant for Cluster {cluster_id} due to generation error.")


print("\n--- Recommending Movies Based on Representants ---")
all_recommended_indices = set()

for cluster_id, rep_emb in representant_embeddings_dict.items():
    similarities = cosine_similarity(rep_emb.reshape(1, -1), candidate_embeddings)[0]
    n_recommendations = 1

    closest_indices = np.argsort(similarities)[::-1][:n_recommendations]
    all_recommended_indices.update(closest_indices)

    print(f"\nTop {n_recommendations} recommendation(s) for Representant {cluster_id} ({representants[cluster_id].genres}):")
    for i, index in enumerate(closest_indices):
    
        if 0 <= index < len(movies):
            closest_movie = movies[index]
            print(f"  {i+1}. {closest_movie['title']} (Similarity: {similarities[index]:.4f})")
        else:
             print(f"  {i+1}. Error: Recommended index {index} out of bounds.")


print("\n--- Visualizing Clusters, Representants, and Candidate Movies (UMAP reduction on ALL points) ---")

num_user = len(user_embeddings)
num_candidate = len(candidate_embeddings)
num_representants = len(representant_embeddings_list)

if num_user < 1:
    print("No user data points for visualization.")
else:

    embeddings_to_combine = [user_embeddings, candidate_embeddings]
    if num_representants > 0:
        embeddings_to_combine.append(np.array(representant_embeddings_list))


    combined_embeddings = np.vstack(embeddings_to_combine)
    print(f"Combined embeddings shape for UMAP: {combined_embeddings.shape}")


    reducer = umap.UMAP(
    
        n_neighbors=max(2, min(15, combined_embeddings.shape[0] - 1)),
        n_components=2,
        metric='euclidean',
        random_state=42,
        min_dist=0.2,
    
    )

    print("Fitting UMAP on combined embeddings...")
    combined_embeddings_2d = reducer.fit_transform(combined_embeddings)
    print("UMAP fitting complete.")


    plt.figure(figsize=(14, 10))
    scatter_handles = []


    color_map = {
        -1: ('grey', 'Noise'), 0: ('blue', 'Animation'), 1: ('red', 'Sci-Fi/Action'),
        2: ('green', 'Drama/Bio'), 3: ('purple', 'Cluster 3'), 4: ('orange', 'Cluster 4'),
    }
    default_cluster_color = 'cyan'
    cluster_names = { 0: "Animation/Family", 1: "Action/Sci-Fi", 2: "Drama/Biography" }

    start_idx_cand = num_user
    end_idx_cand = num_user + num_candidate
    for j in range(num_candidate):
        plot_idx = start_idx_cand + j
        is_recommended = j in all_recommended_indices
        x, y = combined_embeddings_2d[plot_idx]
        plt.scatter(
            x, y,
            color='lightgrey' if not is_recommended else 'limegreen',
            marker='s' if is_recommended else 'x',
            s=70 if is_recommended else 30,
            alpha=0.8 if is_recommended else 0.5,
            edgecolor='black' if is_recommended else 'none',
            label=None
        )
        plt.text(x + 0.01, y + 0.01, movies[j]['title'], fontsize=6, alpha=0.6,
                 color='black' if not is_recommended else 'darkgreen',
                 fontweight='bold' if is_recommended else 'normal')



    seen_labels = set()
    for i in range(num_user):
        plot_idx = i
        x, y = combined_embeddings_2d[plot_idx]
        label = cluster_labels[i]
        color, _ = color_map.get(label, (default_cluster_color, f'Cluster {label}'))
        cluster_name = cluster_names.get(label, f'Cluster {label}') if label != -1 else "Noise"
        handle = plt.scatter(
            x, y,
            color=color,
            alpha=0.9 if label != -1 else 0.4,
            s=60 if label != -1 else 35,
            marker='o',
            edgecolor='black',
            label=cluster_name if label not in seen_labels else None
        )
        if label not in seen_labels:
             scatter_handles.append(handle)
             seen_labels.add(label)
        plt.text(x + 0.01, y + 0.01, user_preferred_movies[i]['title'],
                 fontsize=8, alpha=0.9, weight='bold', color=color)




    start_idx_rep = num_user + num_candidate
    if num_representants > 0:
        for i in range(num_representants):
            plot_idx = start_idx_rep + i
            cluster_id = representant_cluster_ids_list[i]
            x, y = combined_embeddings_2d[plot_idx]
            color, _ = color_map.get(cluster_id, (default_cluster_color, f'Cluster {cluster_id}'))
            rep_handle = plt.scatter(
                x, y,
                color=color,
                marker='*',
                s=300,
                edgecolor='black',
                label=f'Representant {cluster_id} ({cluster_names.get(cluster_id, "")})'
            )
        
        
            is_new_handle_type = True
            for existing_handle in scatter_handles:
                if (existing_handle.get_label() == rep_handle.get_label() and
                    existing_handle.get_marker() == rep_handle.get_marker()):
                    is_new_handle_type = False
                    break
            if is_new_handle_type:
                 scatter_handles.append(rep_handle)


        
            plt.text(x + 0.01, y + 0.01,
                     f"R{cluster_id}: {representants[cluster_id].genres}", fontsize=10, weight='bold', color='black',
                     bbox=dict(facecolor='white', alpha=0.5, pad=0.1, edgecolor='none'))


    if num_candidate > 0:
        if all_recommended_indices:
             rec_handle = plt.scatter([], [], color='limegreen', marker='s', s=70, edgecolor='black', label='Recommended Candidate')
            
             if not any(h.get_label() == rec_handle.get_label() for h in scatter_handles):
                 scatter_handles.append(rec_handle)
        if len(all_recommended_indices) < num_candidate:
             other_handle = plt.scatter([], [], color='lightgrey', marker='x', s=30, alpha=0.6, label='Other Candidate')
             if not any(h.get_label() == other_handle.get_label() for h in scatter_handles):
                 scatter_handles.append(other_handle)

    plt.title('User Preferences, Clusters, Representants & Candidates (Combined UMAP Projection)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(handles=scatter_handles, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()