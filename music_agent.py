import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ---------- Load everything we built in previous steps ----------

df = pd.read_csv("tracks_with_clusters.csv")
feature_columns = ["tempo", "energy", "brightness", "noisiness", "tonality"]

# Recreate the scaler (must match training exactly)
scaler = StandardScaler()
scaler.fit(df[feature_columns].values)

# Recreate the neural network structure (must match train_model.py)
class MusicMoodClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        return self.network(x)

num_classes = df["cluster"].nunique()
model_nn = MusicMoodClassifier(input_size=len(feature_columns), num_classes=num_classes)
model_nn.load_state_dict(torch.load("music_mood_model.pth"))
model_nn.eval()

# Load the embedding model and connect to the existing vector database
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="music_tracks")

# Human-readable labels for each cluster, based on their average features
cluster_profiles = df.groupby("cluster")[feature_columns].mean()

def describe_cluster(cluster_id):
    row = cluster_profiles.loc[cluster_id]
    tempo_desc = "fast" if row["tempo"] > 120 else "slow" if row["tempo"] < 90 else "medium tempo"
    energy_desc = "high energy" if row["energy"] > 0.2 else "low energy"
    return f"{tempo_desc}, {energy_desc}"


# ---------- The Agent: decides which tool to use ----------

# Simple keyword-based rules (no LLM needed - fully free and offline)
RECOMMEND_KEYWORDS = ["recommend", "suggest", "give me a song", "play something",
                      "energetic", "fast", "calm", "relaxing", "upbeat", "slow"]

def agent_respond(user_query):
    query_lower = user_query.lower()

    # Rule: if the query matches simple recommendation keywords -> use the trained model logic (cluster-based)
    uses_recommendation_keywords = any(kw in query_lower for kw in RECOMMEND_KEYWORDS)

    if uses_recommendation_keywords:
        print(f"[Agent] Detected a recommendation request -> using cluster/model-based lookup")
        return recommend_by_mood(user_query)
    else:
        print(f"[Agent] Detected a descriptive/semantic query -> using RAG search")
        return semantic_search(user_query)


def recommend_by_mood(query):
    """Pick a cluster whose profile best matches simple keywords in the query, then sample tracks from it."""
    query_lower = query.lower()

    if "energetic" in query_lower or "fast" in query_lower or "upbeat" in query_lower:
        target_cluster = cluster_profiles["energy"].idxmax()
    elif "calm" in query_lower or "relaxing" in query_lower or "slow" in query_lower:
        target_cluster = cluster_profiles["energy"].idxmin()
    else:
        target_cluster = df["cluster"].mode()[0]  # fallback: most common cluster

    matches = df[df["cluster"] == target_cluster][["name", "artist_name"]].head(3)
    profile = describe_cluster(target_cluster)

    result = f"Based on mood matching (cluster profile: {profile}):\n"
    for _, row in matches.iterrows():
        result += f"  - {row['name']} by {row['artist_name']}\n"
    return result


def semantic_search(query, n_results=3):
    """Search the vector database for tracks whose description matches the query semantically."""
    query_embedding = embed_model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    result = "Based on semantic similarity search:\n"
    for meta in results["metadatas"][0]:
        result += f"  - {meta['name']} by {meta['artist_name']}\n"
    return result


# ---------- Try it out ----------

if __name__ == "__main__":
    test_queries = [
        "recommend me something energetic",
        "I want a calm and relaxing track",
        "something that feels nostalgic and warm",
        "a song for a rainy afternoon",
    ]

    for q in test_queries:
        print(f"\n💬 User: {q}")
        response = agent_respond(q)
        print(response)