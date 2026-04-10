# utils/dataset.py
# Converts tokenized text into input-target pairs for next-token prediction.

import torch
from torch.utils.data import Dataset, DataLoader

class LanguageModelDataset(Dataset):
    """
    PyTorch Dataset for language modeling (next-token prediction).

    Given a long sequence of token IDs, this slices it into fixed-length
    windows. For each window:
        input  = tokens[i : i + seq_len]
        target = tokens[i+1 : i + seq_len + 1]

    This teaches the model: given these N tokens, predict the next N tokens.
    """

    def __init__(self, token_ids, seq_len=32):
        """
        Args:
            token_ids (list[int]): Full encoded token sequence.
            seq_len (int): Length of each training window.
        """
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        # Number of possible windows
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # Input: tokens at positions idx to idx+seq_len-1
        x = self.data[idx : idx + self.seq_len]
        # Target: tokens shifted by 1 (next-token prediction)
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_dataloader(token_ids, seq_len=32, batch_size=16, shuffle=True):
    """
    Wraps the dataset in a DataLoader for batched training.

    Args:
        token_ids (list[int]): Encoded token IDs.
        seq_len (int): Sequence length per sample.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader
    """
    dataset = LanguageModelDataset(token_ids, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"[Dataset] {len(dataset)} samples | batch_size={batch_size} | seq_len={seq_len}")
    return loader
