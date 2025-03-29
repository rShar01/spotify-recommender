import pandas as pd
import numpy as np
import random

from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import KNNImputer
from tqdm import tqdm

def predict(model, row, mask_idx):
    pred = model.transform(row.to_frame().T)
    return pred[0, mask_idx]

    # votes = {}
    # for pred in nbrs[0]:
    #     row = df.iloc[pred]
    #     for i in range(3000):
    #         if row.iloc[i] == 1:
    #             votes[i] = votes.get(i,0) + 1
    # best_song, best_count = -1, 0
    # for song_idx, count in votes.items():
    #     if count > best_count:
    #         best_song = song_idx
    #         best_count = count
    #     # keep the more popular one
    #     elif count == best_count and song_idx < best_song:
    #         best_song = song_idx
    # return best_song

if __name__ == "__main__":
    p = Path("/data/user_data/rshar/downloads/spotify", "simple_vec")
    np.random.seed(42)

    all_csvs = sorted(list(p.glob("*")))
    print(type(all_csvs))

    train_set = random.sample(all_csvs, int(1000 * 0.50))
    test_set = [x for x in all_csvs if x not in train_set]


    dfs = []
    for path in tqdm(train_set):
        curr_csv = pd.read_csv(path, index_col=0)
        curr_csv.drop(["pid", "collaborative", "num_followers", "num_tracks"], axis=1, inplace=True)
        dfs.append(curr_csv)

    df = pd.concat(dfs)

    # model = NearestNeighbors(n_neighbors=100, n_jobs=-1)
    model = KNNImputer(n_neighbors=5)
    model.fit(df)


    test_df = pd.read_csv(test_set[0], index_col=0)
    test_df.drop(["pid", "collaborative", "num_followers", "num_tracks"], axis=1, inplace=True)

    for i in range(5):
        row = test_df.iloc[i]
        mask_idx = random.randint(0, 2999)
        true_val = row.iloc[mask_idx]
        row.iloc[mask_idx] = np.NaN

        pred = predict(model, row, mask_idx)

        # TODO: handle -1 more gracefully
        #       maybe choose a random index to mask ?
        #       we are doing a weird metric rn cause we're using the "predict label" as part of the input
        #       maybe drop one of the cols and use that as the label?
        print(f"Predict: {pred}\t\t err:{row.iloc[pred] == true_val}")
