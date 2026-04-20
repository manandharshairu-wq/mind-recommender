import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, dim, hidden_dim=200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask=None):
        e = torch.tanh(self.proj(x))
        scores = self.query(e).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        output = torch.nan_to_num(output, nan=0.0)
        return output


class NewsEncoder(nn.Module):
    def __init__(self, embedding_matrix, num_heads=16, head_dim=16, dropout=0.2):
        super().__init__()

        if not torch.is_tensor(embedding_matrix):
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
        else:
            embedding_matrix = embedding_matrix.float()

        embed_dim = embedding_matrix.shape[1]
        attn_dim = num_heads * head_dim

        self.word_embed = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, attn_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.additive_attn = AdditiveAttention(attn_dim)

    def forward(self, title_ids):
        token_mask = (title_ids != 0).long()

        x = self.word_embed(title_ids)
        x = self.dropout(x)
        x = self.proj(x)

        x, _ = self.multihead_attn(x, x, x)
        x = torch.nan_to_num(x, nan=0.0)

        x = self.dropout(x)
        news_vec = self.additive_attn(x, token_mask)
        news_vec = torch.nan_to_num(news_vec, nan=0.0)

        return news_vec