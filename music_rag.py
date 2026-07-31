import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load the data with features and clusters
df = pd.read_csv("tracks_with_clusters.csv")

# Since we don't have lyrics, we create a text description for each track
# based on its audio features. This description is what we'll search against.
def describe_track(row):
    # Convert numeric features into natural language descriptions
    tempo_desc = "fast-paced" if row["tempo"] > 120 else "slow-paced" if row["tempo"] < 90 else "medium-tempo"
    energy_desc = "high-energy" if row["energy"] > 0.2 else "calm and low-energy"
    brightness_desc = "bright and sharp sounding" if row["brightness"] > 2500 else "warm and mellow sounding"

    return (
        f"{row['name']} by {row['artist_name']}. "
        f"This track is {tempo_desc}, {energy_desc}, and {brightness_desc}."
    )

df["description"] = df.apply(describe_track, axis=1)

print("Example descriptions:")
print(df["description"].head(3).to_string(index=False))

# Load a sentence embedding model (this downloads a small pretrained model the first time)
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for all track descriptions
print("Creating embeddings...")
embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)

# Set up a local ChromaDB vector database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="music_tracks")

# Add tracks to the vector database
collection.add(
    ids=[str(i) for i in df.index],
    embeddings=embeddings.tolist(),
    documents=df["description"].tolist(),
    metadatas=df[["name", "artist_name", "tempo", "energy"]].to_dict("records")
)

print(f"\nDone! Added {len(df)} tracks to the vector database.")

# --- Now let's test it with a search query ---
def search_music(query, n_results=3):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results

# Example searches
# Example searches
test_queries = [
    "a calm and relaxing song",
    "something energetic and fast",
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    results = search_music(query)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  - {meta['name']} by {meta['artist_name']}")