import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


path = Path("/data/user_data/rshar/downloads/spotify")
jsons = list(path.glob("data/*"))


with open(jsons[0], "r") as f:
    data = json.load(f)

print(data['info'])
print(data['playlists'][0]['tracks'])


uri_counter = {}
song_data_map = {}
for p in tqdm(jsons):
    with open(p, "r") as f:
        data = json.load(f)
        for playlist in data['playlists']:
            for song in playlist['tracks']:
                if song['track_uri'] in uri_counter:
                    uri_counter[song['track_uri']] += 1
                else:
                    uri_counter[song['track_uri']] = 1
                    song_data_map[song['track_uri']] = song

all_song_data = []
for uri, song_data in song_data_map.items():
    row = {'artist_name': song_data['artist_name'], 'track_uri': song_data['track_uri'], 'artist_uri': song_data['artist_uri'], \
    'track_name': song_data['track_name'], 'album_uri': song_data['album_uri'], 'duration_ms': song_data['duration_ms'], 'album_name': song_data['album_name'], 'count': uri_counter[uri]}
    all_song_data.append(row)


df = pd.DataFrame(all_song_data)
df.to_csv("count_songs.csv")