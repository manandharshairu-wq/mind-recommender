import torch
import torch.nn as nn
from src.news_encoder import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(self, news_dim, num_heads=16, head_dim=16, dropout=0.2):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=news_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.additive_attn = AdditiveAttention(news_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, clicked_news_vecs, mask=None):
        # clicked_news_vecs: (batch, history_len, news_dim)
        x, _ = self.multihead_attn(
            clicked_news_vecs,
            clicked_news_vecs,
            clicked_news_vecs
        )
        x = torch.nan_to_num(x, nan=0.0)
        x = self.dropout(x)

        user_vec = self.additive_attn(x, mask)
        user_vec = torch.nan_to_num(user_vec, nan=0.0)
        return user_vec