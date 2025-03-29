import json
import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

df = pd.read_csv("count_songs.csv")
# X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)

d = 3000
top_df = df.nlargest(d, 'count')
top_df = top_df.reset_index(drop=True)

# we use position to indicate feature here
all_song_uris = list(top_df['track_uri'])
with open(f"{d}_songs_list.txt", "w+") as f:
    f.write(str(all_song_uris))


path = Path("/data/user_data/rshar/downloads/spotify")
save_dir = Path(path, "simple_vec")
save_dir.mkdir(parents=True, exist_ok=True)
jsons = list(path.glob("data/*"))


with open(jsons[0], "r") as f:
    data = json.load(f)

print(data['info'])
print(data['playlists'][0]['tracks'])

# playlist metadata:   "collaborative", "duration_ms", "modified_at", "name", 
# "num_albums", "num_artists", "num_edits", "num_followers", "num_tracks", "pid", "tracks"

# keep: collaborative, num_followers, pid, num_tracks

for p in tqdm(jsons):
    rows = []
    with open(p, "r") as f:
        data = json.load(f)
        for playlist in data['playlists']:
            row = dict.fromkeys(range(d), 0)
            row['pid'] = playlist['pid']
            row['collaborative'] = playlist['collaborative']
            row['num_followers'] = playlist['num_followers']
            row['num_tracks'] = playlist['num_tracks']
            for song in playlist['tracks']:
                curr_uri = song['track_uri']
                if curr_uri in all_song_uris:
                    row[all_song_uris.index(curr_uri)] = 1

            rows.append(row)
    df = pd.DataFrame(rows)
    save_path = Path(save_dir, p.name)
    df.to_csv(Path(save_dir, p.name))
    print(f"Saved to {save_path.resolve()}")