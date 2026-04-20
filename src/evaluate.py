import torch
import numpy as np
from sklearn.metrics import roc_auc_score


# -------------------------
# DCG / nDCG
# -------------------------
def dcg_score(y_true, y_score, k):
    order = np.argsort(y_score)[::-1][:k]
    gains = np.array(y_true)[order]
    discounts = np.log2(np.arange(len(gains)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k):
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    return dcg_score(y_true, y_score, k) / best


# -------------------------
# MRR
# -------------------------
def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_sorted = np.array(y_true)[order]

    for i, val in enumerate(y_sorted):
        if val == 1:
            return 1.0 / (i + 1)

    return 0.0


# -------------------------
# MAIN EVALUATION FUNCTION
# -------------------------
def evaluate(model, dataloader, device):

    model.eval()

    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []

    with torch.no_grad():
        for batch in dataloader:

            history = batch['history'].to(device)
            candidates = batch['candidates'].to(device)
            labels = batch['labels'].cpu().numpy()   # (B, K+1)
            hist_mask = batch['hist_mask'].to(device)

            scores = model(history, candidates, hist_mask)
            scores = scores.cpu().numpy()           # (B, K+1)

            for i in range(len(labels)):

                y_true = labels[i]
                y_score = scores[i]

                # skip bad cases (just in case)
                if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
                    continue

                # AUC
                aucs.append(roc_auc_score(y_true, y_score))

                # MRR
                mrrs.append(mrr_score(y_true, y_score))

                # nDCG
                ndcg5s.append(ndcg_score(y_true, y_score, k=5))
                ndcg10s.append(ndcg_score(y_true, y_score, k=10))

    results = {
        "AUC": np.mean(aucs),
        "MRR": np.mean(mrrs),
        "nDCG@5": np.mean(ndcg5s),
        "nDCG@10": np.mean(ndcg10s)
    }

    print("\nEvaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results