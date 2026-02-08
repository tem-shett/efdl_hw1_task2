from typing import Optional

import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
import random

from .transformer import generate_square_subsequent_mask


MAX_LENGTH = 640

def retrieve_parts(data_path):
    parts = []
    for filename in os.listdir(data_path):
        if not filename.startswith('train'):
            continue
        with open(os.path.join(data_path, filename), 'r') as f:
            parts += [p.strip() for p in f.read().split('\n') if 0 < len(p.strip())]

    random.seed(179)
    random.shuffle(parts)
    return parts

class WikiTextDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.tokenizer.model_max_length = 640
        data = retrieve_parts(data_path)
        self.input_ids = []
        for s in data:
            token_ids = self.tokenizer.encode(s, add_special_tokens=False)
            if 0 < len(token_ids) <= MAX_LENGTH:
                self.input_ids.append(token_ids)
        self.input_ids = self.input_ids[:30000]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx]

class BrainDataset(WikiTextDataset):
    pass


class BigBrainDataset(WikiTextDataset):
    pass


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, n_bins: int = 1):
        pass

    def __getitem__(self, idx: int):
        pass


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __getitem__(self, idx: int):
        pass


def collate_fn_brain(
    batch: list[list[int]], max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    B = len(batch)
    T = max_length - 1
    src_ids = []
    tgt_ids = []
    key_padding_mask = torch.zeros((B, T), dtype=torch.bool)
    for i, tokens in enumerate(batch):
        src_ids.append(tokens[:-1] + [0] * (T - (len(tokens) - 1)))
        tgt_ids.append(tokens[1:] + [0] * (T - len(tokens) - 1))
        key_padding_mask[i, len(tokens) - 1:] = True
    
    causal_mask = generate_square_subsequent_mask(T)
    return torch.tensor(src_ids), torch.tensor(tgt_ids), causal_mask, key_padding_mask


class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, batch_size: int, max_length: Optional[int] = MAX_LENGTH):
        pass

    def __len__(self):
        pass

    def __iter__(self):
        pass
