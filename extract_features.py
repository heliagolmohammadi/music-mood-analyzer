import pandas as pd
import requests
import os
import librosa
import numpy as np

# Settings
NUM_TRACKS = 50
AUDIO_FOLDER = "audio_files"
CSV_INPUT = "jamendo_tracks.csv"
CSV_OUTPUT = "tracks_with_features.csv"

# Create a folder to store downloaded audio files
os.makedirs(AUDIO_FOLDER, exist_ok=True)

def download_audio(url, filepath):
    """Download an mp3 file from a URL and save it locally."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        return False

def extract_audio_features(filepath):
    """Extract basic audio features from a track using librosa."""
    try:
        y, sr = librosa.load(filepath, sr=22050, duration=30)  # first 30 seconds is enough

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))  # energy/loudness
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # brightness
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))  # noisiness
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))  # tonal content

        return {
            "tempo": float(tempo),
            "energy": float(rms),
            "brightness": float(spectral_centroid),
            "noisiness": float(zero_crossing_rate),
            "tonality": float(chroma),
        }
    except Exception as e:
        print(f"  Failed to extract features: {e}")
        return None

if __name__ == "__main__":
    df = pd.read_csv(CSV_INPUT)
    df = df.head(NUM_TRACKS).copy()

    results = []

    for i, row in df.iterrows():
        track_id = row["id"]
        audio_url = row["audio"]
        filename = os.path.join(AUDIO_FOLDER, f"{track_id}.mp3")

        print(f"[{i+1}/{len(df)}] {row['name']} - {row['artist_name']}")

        # Download the audio file if not already downloaded
        if not os.path.exists(filename):
            success = download_audio(audio_url, filename)
            if not success:
                continue
        else:
            print("  Already downloaded, skipping download.")

        # Extract features
        features = extract_audio_features(filename)
        if features is None:
            continue

        row_data = row.to_dict()
        row_data.update(features)
        results.append(row_data)

    result_df = pd.DataFrame(results)
    result_df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\nDone! Extracted features for {len(result_df)} tracks.")
    print(f"Saved to {CSV_OUTPUT}")
    print(result_df[["name", "artist_name", "tempo", "energy", "brightness"]].head())