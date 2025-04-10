from typing import Optional

import torch
from torch import Tensor, nn
import numpy as np


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int] = None,
        cond_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        time_varying: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.time_varying = time_varying

        if out_dim is None:
            out_dim = dim

        layers = [
            nn.Linear(
                dim
                + (1 if time_varying else 0)
                + (cond_dim if cond_dim is not None else 0),
                hidden_dim,
            ),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        ]

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden_dim))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, cond: Tensor = None):
        if cond is not None:
            x = torch.cat([x, cond], dim=-1)
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)]


class TransformerVecField(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conditional_static_dim: int,
        transformer_encoder_n_heads: int,
        transformer_encoder_dim_forward: int,
        transformer_encoding_n_layers: int,
        learned_pe: bool = False,
        learned_time_embedding: bool = False,
        time_embedding_size: int = 32,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.conditional_static_dim = conditional_static_dim
        self.transformer_encoder_n_heads = transformer_encoder_n_heads
        self.transformer_encoder_dim_forward = transformer_encoder_dim_forward
        self.transformer_encoding_n_layers = transformer_encoding_n_layers
        self.learned_pe = learned_pe
        self.learned_time_embedding = learned_time_embedding
        self.time_embedding_size = time_embedding_size
        self.dropout = nn.Dropout(0.1)

        if learned_time_embedding:
            self.flow_time_embedder = nn.Sequential(
                nn.Linear(1, time_embedding_size),
                nn.SiLU(),
                nn.Linear(time_embedding_size, time_embedding_size),
            )

        if learned_pe:
            self.pe = nn.Sequential(
                nn.Linear(1, time_embedding_size),
                nn.SiLU(),
                nn.Linear(time_embedding_size, time_embedding_size),
            )
        else:
            self.pe = PositionalEncoding(input_dim)

    
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim + time_embedding_size,
                nhead=transformer_encoder_n_heads,
                dim_feedforward=transformer_encoder_dim_forward,
                batch_first=True,
            ),
            num_layers=transformer_encoding_n_layers,
        )

        self.fc_output = nn.Linear(
            input_dim + time_embedding_size + conditional_static_dim, input_dim
        )

    @staticmethod
    def get_sinusoidal_time_embedding(t: Tensor, embedding_dim: int) -> Tensor:
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb).type_as(t)
        emb = t.unsqueeze(1) * emb
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, t: Tensor, x: Tensor, conditional_static: Tensor, seq_sizes: Tensor) -> Tensor:

        # x: (batch_size, num_time_steps, input_dim)
        # t: (batch_size)
        # conditional_static: (batch_size, conditional_static_dim)
        # seq_sizes: (batch_size)

        # input embeddings to transformer will be the following:
        # concat(x + positional encoding, flow time embedding)
        if not t.ndim:
            t = t.repeat(x.size(0))
        if self.learned_time_embedding:
            t = self.flow_time_embedder(t.unsqueeze(1))
        else:
            t = self.get_sinusoidal_time_embedding(t, self.time_embedding_size)
        t = t.unsqueeze(1).expand(-1, x.size(1), -1)

        # positional encoding
        if self.learned_pe:
            pe = self.pe(torch.arange(x.size(1)).unsqueeze(1).type_as(x))
            pe = pe.unsqueeze(0).repeat(x.size(0), 1, 1)
        else:
            pe = self.pe(x)
            pe = pe.repeat(x.size(0), 1, 1)

        max_seq_len = x.size(1)
        x = self.dropout(
            torch.concat(
                (
                    x + pe,
                    t,
                ),
                dim=-1,
            )
        )

        batch_size = x.size(0)
        # Create position indices for each sequence
        pos_indices = torch.arange(max_seq_len).to(x.device).unsqueeze(0).expand(batch_size, -1)

        key_padding_mask = pos_indices >= seq_sizes.unsqueeze(1)
        key_padding_mask = key_padding_mask.float()
        key_padding_mask = key_padding_mask.masked_fill(key_padding_mask == 1.0, float('-inf'))

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        x = torch.concat(
            (x, conditional_static.unsqueeze(1).repeat(1, x.size(1), 1)), dim=-1
        )
        x = self.fc_output(self.dropout(x))

        return x
