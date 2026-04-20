import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, samples, max_history=50, max_title_len=30):
        self.samples = samples
        self.max_history = max_history
        self.max_title_len = max_title_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return {
            "history": sample["history"],
            "candidates": sample["candidates"],
            "labels": sample["labels"]   # list like [1,0,0,0,...]
        }


def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))


def collate_fn(batch):
    histories = []
    candidates = []
    labels = []
    hist_masks = []

    for sample in batch:

        # ------------------------
        # HISTORY
        # ------------------------
        history = sample["history"]

        padded_history = []
        for news in history[:50]:
            padded_news = pad_sequence(news, 30)
            padded_history.append(padded_news)

        while len(padded_history) < 50:
            padded_history.append([0] * 30)

        padded_history = torch.tensor(padded_history, dtype=torch.long)
        histories.append(padded_history)

        # ------------------------
        # HISTORY MASK
        # ------------------------
        mask = (padded_history.sum(dim=1) != 0).float()
        hist_masks.append(mask)

        # ------------------------
        # CANDIDATES
        # ------------------------
        cand = sample["candidates"]

        padded_cand = []
        for news in cand:
            padded_news = pad_sequence(news, 30)
            padded_cand.append(padded_news)

        padded_cand = torch.tensor(padded_cand, dtype=torch.long)
        candidates.append(padded_cand)

        # ------------------------
        # LABELS 
        # ------------------------
        labels.append(torch.tensor(sample["labels"], dtype=torch.float))

    # ------------------------
    # STACK EVERYTHING
    # ------------------------
    histories = torch.stack(histories)        # (B, 50, 30)
    candidates = torch.stack(candidates)      # (B, K+1, 30)
    hist_masks = torch.stack(hist_masks)      # (B, 50)
    labels = torch.stack(labels)              # (B, K+1)

    return {
        "history": histories,
        "candidates": candidates,
        "labels": labels,
        "hist_mask": hist_masks
    }