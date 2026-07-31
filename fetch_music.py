import requests
import pandas as pd
import time

CLIENT_ID = "85586e56"  # your Jamendo Client ID
BASE_URL = "https://api.jamendo.com/v3.0/tracks/"

def fetch_tracks(limit=200, batch_size=50):
    all_tracks = []
    offset = 0
    
    while len(all_tracks) < limit:
        params = {
            "client_id": CLIENT_ID,
            "format": "json",
            "limit": batch_size,
            "offset": offset,
            "audioformat": "mp32",  # audio file format
            "include": "musicinfo",  # also fetch genre/tag info
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if data["headers"]["status"] != "success":
            print("Error fetching data:", data["headers"])
            break
        
        tracks = data["results"]
        if not tracks:
            print("No more tracks available.")
            break
        
        all_tracks.extend(tracks)
        offset += batch_size
        print(f"Fetched {len(all_tracks)} tracks so far...")
        
        time.sleep(0.5)  # small delay to avoid hammering the server
    
    return all_tracks[:limit]

if __name__ == "__main__":
    tracks = fetch_tracks(limit=200)
    
    # Convert to a DataFrame and save as CSV
    df = pd.DataFrame(tracks)
    df.to_csv("jamendo_tracks.csv", index=False, encoding="utf-8-sig")
    
    print(f"\nDone! {len(df)} tracks saved to jamendo_tracks.csv")
    print(df[["name", "artist_name", "duration", "audio"]].head())