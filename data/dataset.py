import random
import json

import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from data.augmentation import pitch_shift, change_speed

def normalize_time_features(time_feature: np.ndarray, mean: float, std: float):
    log2_x = np.log2(time_feature + 1e-8)

    return (log2_x - mean) / std

def denormalize_time_features(time_feature: np.ndarray, mean: float, std: float):
    time_feature = std * time_feature + mean

    return 2 ** time_feature - 1e-8


class MidiDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        pitch_shift_probability: float = 0.0,
        time_stretch_probability: float = 0.0,
        use_dstart_log_normalization: bool = False,
    ):
        super().__init__()

        self.dataset = dataset.with_format("numpy")

        self.pitch_shift_probability = pitch_shift_probability
        self.time_stretch_probability = time_stretch_probability

        self.use_dstart_log_normalization = use_dstart_log_normalization

        if use_dstart_log_normalization:
            time_normalization_features = json.load(open("artifacts/time_features.json"))

            self.mean_dstart = time_normalization_features["mean_dstart"]
            self.std_dstart = time_normalization_features["std_dstart"]

    def __rich_repr__(self):
        yield "MidiDataset"
        yield "size", len(self)
        yield "pitch_shift_prob", self.pitch_shift_probability
        yield "time_stretch_prob", self.time_stretch_probability

    def __len__(self):
        return len(self.dataset)

    def apply_augmentation(self, record: dict):
        # shift pitch augmentation
        if random.random() < self.pitch_shift_probability:
            shift = 7
            record["pitch"] = pitch_shift(pitch=record["pitch"], shift_threshold=shift)

        # change tempo augmentation
        if random.random() < self.time_stretch_probability:
            record["dstart"], record["duration"] = change_speed(dstart=record["dstart"], duration=record["duration"])

        return record

    def __getitem__(self, index: int) -> dict:
        record = self.dataset[index]

        # sanity check, replace NaN with 0
        if np.any(np.isnan(record["dstart"])):
            record["dstart"] = np.nan_to_num(record["dstart"], copy=False)

        record = self.apply_augmentation(record)
        
        # clip outliers
        record["dstart"] = np.clip(record["dstart"], 0.0, 5.0)

        # apply log2 normalization for dstart
        if self.use_dstart_log_normalization:
            record["dstart"] = normalize_time_features(record["dstart"], self.mean_dstart, self.std_dstart)

        data = {
            "filename": record["midi_filename"],
            "source": record["source"],
            "pitch": torch.tensor(record["pitch"], dtype=torch.long) - 21,
            "velocity": (torch.tensor(record["velocity"], dtype=torch.float) / 64) - 1,
            "dstart": torch.tensor(record["dstart"], dtype=torch.float),
            "duration": torch.tensor(record["duration"], dtype=torch.float).clip(0.0, 5.0),
        }

        return data
