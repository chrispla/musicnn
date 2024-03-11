config = {
    # training
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "lr_patience": 10,
    "epochs": 1000,
    # input
    "hop_length": 256,
    "n_fft": 512,
    "n_mels": 96,
    "sr": 16000,
    # dataset
    "filetype": "wav",
    "items_per_track": 10,
    "only_from_tag": None,
    "tracks_per_genre": None,
    "random_percentage": None,
}
