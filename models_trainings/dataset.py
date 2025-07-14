# models_trainings/dataset.py

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, seq_len, pred_len, target_columns=None):
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        df = df.select_dtypes(include=["number"]).dropna()

        if target_columns:
            self.target_cols = [col for col in df.columns if col in target_columns]
        else:
            self.target_cols = df.columns[1:].tolist()  # default to all but first

        data = df[self.target_cols].values
        self.sequences, self.labels = [], []
        for i in range(len(data) - seq_len - pred_len):
            self.sequences.append(data[i:i+seq_len])
            self.labels.append(data[i+seq_len:i+seq_len+pred_len])
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
