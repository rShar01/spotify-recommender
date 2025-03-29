import pandas as pd
import json
from pathlib import Path


df = pd.read_csv("playlist_data_minus_tracks.csv")
print(df.columns)


print(df[["num_tracks", "num_albums", "num_followers", "num_edits", "duration_ms", "num_artists"]].corr())