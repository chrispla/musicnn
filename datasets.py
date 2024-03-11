"""Dataset classes. Modified from github.com/chrispla/mir_ref"""

import csv
import json
import os
import zipfile
from pathlib import Path

import librosa
import numpy as np
import torch
import wget
from torch.utils.data import Dataset
from tqdm import tqdm


class MTAT(Dataset):
    def __init__(
        self,
        root,
        download=True,
        pre_train=True,
    ):
        """MagnaTagATune dataset class.
        Args:
            items_per_track: int, number of items to divide by
            only_from_tag: tag name, where only tracks with that tag are going to be
                           used for training
            tracks_per_genre: number of tracks to use from the top 12 genres for
                              training
            random_percentage: percentage of tracks to use, randomly selected with a
                               seed of 0. Out of 100 (e.g. <22.5>)
        """
        super().__init__()

        # open json config
        with open("TEMP_mtat_config.json", "r") as f:
            mtat_config = json.load(f)

        self.root = os.path.join(root, "music")
        self.filetype = mtat_config["filetype"]
        self.items_per_track = mtat_config["items_per_track"]

        # audio processing
        self.sample_rate = mtat_config["sample_rate"]
        self.min_track_len_seconds = 29.124036281179137
        self.min_track_len_frames = int(self.min_track_len_seconds * self.sample_rate)
        self.item_len_seconds = self.min_track_len_seconds / self.items_per_track
        self.item_len_frames = int(self.min_track_len_frames / self.items_per_track)

        # subset parameters
        self.only_from_tag = mtat_config["only_from_tag"]
        self.tracks_per_genre = mtat_config["tracks_per_genre"]
        self.random_percentage = mtat_config["random_percentage"]

        # dataset
        self.track_ids = []
        self.audio_paths = {}
        self.labels = {}
        self.encoded_labels = {}

        # mel spectrogram parameters
        self.mel_hop_length = mtat_config["mel_hop_length"]
        self.mel_n_fft = mtat_config["mel_n_fft"]
        self.mel_n_mels = mtat_config["mel_n_mels"]

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if download:
            self.download_audio()
            self.download_metadata()

        self.load_track_ids()
        self.load_audio_paths()
        self.load_labels()
        self.load_encoded_labels()

    def download_audio(self):
        # make data dir if it doesn't exist or if it exists but is empty
        if os.path.exists(os.path.join(self.root, "MTAT", "audio")) and (
            len(os.listdir(os.path.join(self.root, "MTAT", "audio"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Dataset already exists in '{self.root}'."
                + "Skipping audio download.",
                stacklevel=2,
            )
            self.download_metadata()
            return
        (Path(self.root) / "MTAT" / "audio").mkdir(parents=True, exist_ok=True)

        print(f"Downloading MagnaTagATune to {self.root}...")
        for i in tqdm(["001", "002", "003"]):
            wget.download(
                url=f"https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.{i}",
                out=os.path.join(self.root, "MTAT/", "audio"),
            )

        archive_dir = os.path.join(self.root, "MTAT", "audio")

        # Combine the split archive files into a single file
        with open(os.path.join(archive_dir, "mp3.zip"), "wb") as f:
            for i in ["001", "002", "003"]:
                with open(
                    os.path.join(archive_dir, f"mp3.zip.{i}"),
                    "rb",
                ) as part:
                    f.write(part.read())

        # Extract the contents of the archive
        with zipfile.ZipFile(os.path.join(archive_dir, "mp3.zip"), "r") as zip_ref:
            zip_ref.extractall()

        # Remove zips
        for i in ["", ".001", ".002", ".003"]:
            os.remove(os.path.join(archive_dir, f"mp3.zip{i}"))

    def download_metadata(self):
        if os.path.exists(os.path.join(self.root, "MTAT", "metadata")) and (
            len(os.listdir(os.path.join(self.root, "MTAT", "metadata"))) != 0
        ):
            import warnings

            warnings.warn(
                f"Metadata for dataset already exists in '{self.root}'."
                + "Skipping metadata download.",
                stacklevel=2,
            )
            return
        (Path(self.root) / "metadata").mkdir(parents=True, exist_ok=True)

        urls = [
            # annotations
            "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv",
            # train, validation, and test splits from Won et al. 2020
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy",
            "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy",
        ]
        for url in urls:
            wget.download(
                url=url,
                out=os.path.join(self.root, "MTAT", "metadata/"),
            )

    def load_track_ids(self):
        with open(
            os.path.join(self.root, "MTAT", "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.track_ids = [line[0] for line in annotations]
            # manually remove some corrupt files
            try:
                self.track_ids.remove("35644")
                self.track_ids.remove("55753")
                self.track_ids.remove("57881")
            except ValueError:
                pass

        if self.random_percentage:
            self.random_percentage = float(self.random_percentage) / 100.0
            np.random.seed(0)
            self.track_ids = np.random.choice(
                self.track_ids,
                size=int(len(self.track_ids) * self.random_percentage),
                replace=False,
            )

    def load_labels(self):
        # get the list of top 50 tags used in Minz Won et al. 2020
        tags = np.load(os.path.join(self.root, "MTAT", "metadata", "tags.npy"))

        with open(
            os.path.join(self.root, "MTAT", "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            annotations_header = next(annotations)
            self.labels = {
                line[0]: [
                    annotations_header[j]
                    for j in range(1, len(line) - 1)
                    # only add the tag if it's in the tags list
                    if line[j] == "1" and annotations_header[j] in tags
                ]
                for line in annotations
                # this is a slow way to do it, temporary fix for
                # some corrupt mp3s
                if line[0] in self.track_ids
            }

        if self.only_from_tag:
            self.track_ids = [
                t_id
                for t_id in self.track_ids
                if self.only_from_tag in self.labels[t_id]
            ]

        if self.tracks_per_genre:
            # only keep track_ids for 100 tracks of each genre
            new_track_ids = []
            genres = [
                "classical",
                "techno",
                "electronic",
                "rock",
                "indian",
                "opera",
                "pop",
                "classic",
                "new age",
                "dance",
                "country",
                "metal",
            ]
            counts = {k: 0 for k in genres}
            for t_id in self.track_ids:
                for label in self.labels[t_id]:
                    if label in genres:
                        if counts[label] < self.tracks_per_genre:
                            new_track_ids.append(t_id)
                            counts[label] += 1
            self.track_ids = new_track_ids

    def load_audio_paths(self):
        with open(
            os.path.join(self.root, "MTAT", "metadata", "annotations_final.csv"), "r"
        ) as f:
            annotations = csv.reader(f, delimiter="\t")
            next(annotations)  # skip header
            self.audio_paths = {
                line[0]: os.path.join(self.root, "MTAT", "audio", line[-1])
                for line in annotations
                # this is a slow way to do it, temporary fix for some corrupt mp3s
                if line[0] in self.track_ids
            }
            if self.filetype == "wav":
                for k, v in self.audio_paths.items():
                    self.audio_paths[k] = v.replace(".mp3", ".wav")

    def load_encoded_labels(self):
        # get lists of track_ids and labels (corresponding indices)
        track_ids = list(self.labels.keys())
        labels_list = list(self.labels.values())

        from sklearn.preprocessing import MultiLabelBinarizer

        # fit label encoder on all tracks and labels
        self.label_encoder = MultiLabelBinarizer()
        self.label_encoder.fit(labels_list)
        # get encoded labels
        encoded_labels_list = self.label_encoder.transform(labels_list)
        self.encoded_labels = {
            track_ids[i]: encoded_labels_list[i] for i in range(len(track_ids))
        }

    def __len__(self):
        return len(self.track_ids) * self.items_per_track

    def __getitem__(self, idx):
        # get t_id given self.items_per_track
        t_id = self.track_ids[idx // self.items_per_track]

        # librosa supports loading specific parts of a track (if not mp3)
        # we'll compute which parts in seconds using the within-track
        # index and the item length in seconds
        start = (idx % self.items_per_track) * self.item_len_seconds
        end = start + self.item_len_seconds

        if self.filetype == "wav":
            y, _ = librosa.load(
                self.audio_paths[t_id],
                sr=self.sample_rate,
                offset=start,
                duration=self.item_len_seconds,
            )
        else:
            # we unfortunately need to load the whole file, then slice
            y, _ = librosa.load(
                self.audio_paths[t_id],
                sr=self.sample_rate,
            )
            y = y[int(start * self.sample_rate) : int(end * self.sample_rate)]

        # crop or pad to item_len_frames if needed
        if len(y) < self.item_len_frames:
            y = np.pad(
                y,
                (0, self.item_len_frames - len(y)),
                mode="constant",
            )
        elif len(y) > self.item_len_frames:
            y = y[: self.item_len_frames]

        # we'll compute the mel spectrogram using torchaudio during training
        return torch.FloatTensor(y), torch.LongTensor(self.encoded_labels[t_id])
