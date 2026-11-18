import math
import re
from collections import Counter

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def build_vocabulary(texts: pd.Series, min_frequency: int) -> dict[str, int]:
    token_counts = Counter(token for text in texts for token in tokenize(text))

    vocabulary = [
        token for token, count in token_counts.items() if count >= min_frequency
    ]
    vocabulary.sort(key=lambda token: (-token_counts[token], token))

    token_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}

    for token in vocabulary:
        token_to_id[token] = len(token_to_id)

    return token_to_id


def prepare_text(
    text: str, token_to_id: dict[str, int], max_length: int
) -> tuple[list[int], list[int]]:
    token_ids = [token_to_id.get(token, UNK_ID) for token in tokenize(text)][
        :max_length
    ]

    if not token_ids:
        token_ids = [UNK_ID]

    attention_mask = [1] * len(token_ids)

    padding_length = max_length - len(token_ids)
    token_ids += [PAD_ID] * padding_length
    attention_mask += [0] * padding_length

    return token_ids, attention_mask


class BankIntentDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        token_to_id: dict[str, int],
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.token_to_id = token_to_id
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]

        input_ids, attention_mask = prepare_text(
            text=row["text"], token_to_id=self.token_to_id, max_length=self.max_length
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.label_to_id[row["category"]], dtype=torch.long),
        }


class IntentTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_ID
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=embedding_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape

        positions = (
            torch.arange(sequence_length, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, sequence_length)
        )

        hidden_states = self.token_embedding(input_ids) * math.sqrt(
            self.embedding_dim
        ) + self.position_embedding(positions)

        padding_mask = attention_mask == 0

        encoded = self.encoder(hidden_states, src_key_padding_mask=padding_mask)

        valid_tokens = attention_mask.unsqueeze(-1).type_as(encoded)

        pooled = (encoded * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(
            min=1.0
        )

        return self.classifier(pooled)
