import torch
import torch.nn as nn
import math
from .transformer import PositionalEncoding, generate_square_subsequent_mask


class GPT2LikeModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 640,
        d_model: int = 1024,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.n_heads = n_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout=dropout, max_len=max_length)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, causal_mask: torch.Tensor, key_padding_mask: torch.Tensor):
        B, T = input_ids.shape
        device = input_ids.device

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_emb(x)

        if causal_mask.dim() == 3:
            causal_mask = causal_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1).reshape(-1, T, T)

        x = self.decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )

        logits = self.lm_head(x)
        return logits
