import numpy as np
import requests
from sklearn.cluster import KMeans
from neo4j import GraphDatabase

# Helper functions
def generate_embedding(text, model="qwen2.5-coder:latest", url="http://127.0.0.1:11434/api/generate"):
    """Generate an embedding for the given text using Qwen."""
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": f"Generate an embedding for: {text}"},
            timeout=10
        )
        if response.status_code == 200:
            # Extract the embedding from the response (assuming embedding is returned in `embedding` field)
            json_response = response.json()
            return json_response.get("embedding", [])
        else:
            print(f"Error: Received status code {response.status_code} from Qwen.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Qwen: {e}")
        return []

def fetch_movies(conn):
    """Retrieve all movies with their titles, genres, and plots from Neo4j."""
    query = """
        MATCH (m:Movie)-[:GENRE]->(g:Genre), (m)-[:HAS_PLOT]->(p:Plot)
        RETURN m.Title AS title, g.Type AS genre, p.Description AS plot
    """
    with conn.session() as session:
        results = session.run(query)
        return [
            {"title": record["title"], "genre": record["genre"], "plot": record["plot"], "movie_id": record["movie_id"]}
            for record in results
        ]

def store_embeddings(conn, movie_id, embedding):
    """Store the embedding for a movie in Neo4j."""
    query = """
    MATCH (m:Movie)
    WHERE id(m) = $movie_id
    SET m.embedding = $embedding
    """
    with conn.driver.session() as session:
        session.run(query, movie_id=movie_id, embedding=embedding)

def cluster_movies(embeddings, n_clusters=10):
    """Cluster movies based on their embeddings."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

def display_clusters(movies, labels, top_n=5):
    """Display the top N clusters and their movies."""
    cluster_map = {}
    for movie, label in zip(movies, labels):
        cluster_map.setdefault(label, []).append(movie)

    sorted_clusters = sorted(cluster_map.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\nTop {top_n} Clusters:")
    for cluster_id, cluster_movies in sorted_clusters[:top_n]:
        print(f"\nCluster {cluster_id} (Size: {len(cluster_movies)})")
        for movie in cluster_movies:
            print(f"  - {movie['title']} (Genre: {movie['genre']}, Plot: {movie['plot'][:50]}...)")

# Main script
def main():
    # Neo4j connection setup
    conn = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "cinema123"))

    # Step 1: Fetch movies
    movies = fetch_movies(conn)
    print(f"Retrieved {len(movies)} movies from Neo4j.")

    # Step 2: Generate embeddings
    for movie in movies:
        text = f"{movie['title']} {movie['genre']} {movie['plot']}"
        embedding = generate_embedding(text)
        if embedding:
            store_embeddings(conn, movie["movie_id"], embedding)
            movie["embedding"] = embedding

    # Step 3: Cluster movies
    embeddings = [movie["embedding"] for movie in movies if "embedding" in movie]
    labels, kmeans = cluster_movies(embeddings)

    # Step 4: Display top 5 clusters
    display_clusters(movies, labels)

    # Close Neo4j connection
    conn.close()

if __name__ == "__main__":
    main()
