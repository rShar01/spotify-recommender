import pandas as pd
import json
from pathlib import Path


path = Path("/data/user_data/rshar/downloads/spotify")
jsons = list(path.glob("data/*"))

with open(jsons[0], "r") as f:
    data = json.load(f)

print(data['info'])
print(data['playlists'][0].keys())


rows = []
for file in jsons:
    with open(file, "r") as f:
        data = json.load(f)
    
    for plst in data['playlists']:
        plst.pop('tracks')
        rows.append(plst)
    

df = pd.DataFrame(rows)
df.to_csv("playlist_data_minus_tracks.csv")


