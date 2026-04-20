import torch
import torch.nn as nn
from src.news_encoder import NewsEncoder
from src.user_encoder import UserEncoder


class NRMSModel(nn.Module):
    def __init__(self, embedding_matrix, num_heads=16, head_dim=16, dropout=0.2):
        super().__init__()

        news_dim = num_heads * head_dim

        self.news_encoder = NewsEncoder(
            embedding_matrix=embedding_matrix,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )

        self.user_encoder = UserEncoder(
            news_dim=news_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )

    def forward(self, history_ids, candidate_ids, hist_mask=None):
        # history_ids: (batch, hist_len, title_len)
        # candidate_ids: (batch, num_candidates, title_len)

        batch_size, hist_len, title_len = history_ids.shape
        _, num_candidates, _ = candidate_ids.shape

        # Encode history
        hist_flat = history_ids.view(-1, title_len)
        hist_vecs = self.news_encoder(hist_flat)
        hist_vecs = hist_vecs.view(batch_size, hist_len, -1)

        # Encode candidates
        cand_flat = candidate_ids.view(-1, title_len)
        cand_vecs = self.news_encoder(cand_flat)
        cand_vecs = cand_vecs.view(batch_size, num_candidates, -1)

        # User vector
        user_vec = self.user_encoder(hist_vecs, hist_mask)

        # Dot-product scores
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        scores = torch.nan_to_num(scores, nan=0.0)

        return scores