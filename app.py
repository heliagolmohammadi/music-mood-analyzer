import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ---------- Load everything (same as music_agent.py) ----------

@st.cache_resource
def load_everything():
    df = pd.read_csv("tracks_with_clusters.csv")
    feature_columns = ["tempo", "energy", "brightness", "noisiness", "tonality"]

    scaler = StandardScaler()
    scaler.fit(df[feature_columns].values)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="music_tracks")

    cluster_profiles = df.groupby("cluster")[feature_columns].mean()

    return df, embed_model, collection, cluster_profiles

df, embed_model, collection, cluster_profiles = load_everything()


def describe_cluster(cluster_id):
    row = cluster_profiles.loc[cluster_id]
    tempo_desc = "fast" if row["tempo"] > 120 else "slow" if row["tempo"] < 90 else "medium tempo"
    energy_desc = "high energy" if row["energy"] > 0.2 else "low energy"
    return f"{tempo_desc}, {energy_desc}"


RECOMMEND_KEYWORDS = ["recommend", "suggest", "give me a song", "play something",
                      "energetic", "fast", "calm", "relaxing", "upbeat", "slow"]


def recommend_by_mood(query):
    query_lower = query.lower()
    if "energetic" in query_lower or "fast" in query_lower or "upbeat" in query_lower:
        target_cluster = cluster_profiles["energy"].idxmax()
    elif "calm" in query_lower or "relaxing" in query_lower or "slow" in query_lower:
        target_cluster = cluster_profiles["energy"].idxmin()
    else:
        target_cluster = df["cluster"].mode()[0]

    matches = df[df["cluster"] == target_cluster][["name", "artist_name"]].head(3)
    profile = describe_cluster(target_cluster)
    return matches, f"Mood matching (cluster profile: {profile})"


def semantic_search(query, n_results=3):
    query_embedding = embed_model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
    names = [meta["name"] for meta in results["metadatas"][0]]
    artists = [meta["artist_name"] for meta in results["metadatas"][0]]
    matches = pd.DataFrame({"name": names, "artist_name": artists})
    return matches, "Semantic similarity search"


def agent_respond(user_query):
    query_lower = user_query.lower()
    uses_recommendation_keywords = any(kw in query_lower for kw in RECOMMEND_KEYWORDS)
    if uses_recommendation_keywords:
        return recommend_by_mood(user_query)
    else:
        return semantic_search(user_query)


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Music Mood Analyzer", page_icon="🎵")
st.title("🎵 Music Mood Analyzer")
st.write("Describe the kind of music you're looking for, and the AI agent will find matching tracks.")

user_input = st.text_input(
    "What are you in the mood for?",
    placeholder="e.g. something energetic, or a calm relaxing song..."
)

if st.button("Find music") and user_input:
    with st.spinner("Thinking..."):
        matches, method_used = agent_respond(user_input)

    st.caption(f"Method used: {method_used}")

    if len(matches) == 0:
        st.warning("No matches found.")
    else:
        for _, row in matches.iterrows():
            st.write(f"**{row['name']}** — {row['artist_name']}")

st.divider()
st.caption(f"Dataset: {len(df)} tracks analyzed | Clusters: {df['cluster'].nunique()}")