2 262 292 unique songs
    - Doing an embedding from song_id to vector seems not viable
    - 287 740 unique artists which doesn't help too much -- on avg each artist doing 10 songs
    - Simple: each playlist has 2 262 292 features, one for each song
        - Can do some dim reduction
        - Can do simple similarity algs
    - ModernBERT to embed each song?
    - Ah, should also keep track of the number of times each song appears
        - Gets us the most popular songs for good baseline
        - Lets us keep top n if we want to do some simpler thing

        


