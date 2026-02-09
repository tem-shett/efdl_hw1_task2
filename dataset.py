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
    return parts[:100000]

class WikiTextDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        data = retrieve_parts(data_path)
        self.input_ids = []
        for s in data:
            token_ids = self.tokenizer.encode(s, add_special_tokens=False)
            if 0 < len(token_ids) <= MAX_LENGTH:
                self.input_ids.append(token_ids)
                if len(self.input_ids) >= 30000:
                    break

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx]

class BrainDataset(WikiTextDataset):
    pass


class BigBrainDataset(WikiTextDataset):
    pass


class UltraBigBrainDataset(WikiTextDataset):
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
    key_padding_mask = torch.zeros((B, T), dtype=torch.float)
    for i, tokens in enumerate(batch):
        src_ids.append(tokens[:-1] + [0] * (T - (len(tokens) - 1)))
        tgt_ids.append(tokens[1:] + [0] * (T - (len(tokens) - 1)))
        key_padding_mask[i, len(tokens) - 1:] = True
    
    causal_mask = generate_square_subsequent_mask(T)
    return torch.tensor(src_ids), torch.tensor(tgt_ids), causal_mask, key_padding_mask


def collate_fn_bigbrain(
    batch: list[list[int]], max_length: Optional[int] = MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    B = len(batch)
    T = max(len(el) for el in batch) - 1
    src_ids = []
    tgt_ids = []
    key_padding_mask = torch.zeros((B, T), dtype=torch.float)
    for i, tokens in enumerate(batch):
        src_ids.append(tokens[:-1] + [0] * (T - (len(tokens) - 1)))
        tgt_ids.append(tokens[1:] + [0] * (T - (len(tokens) - 1)))
        key_padding_mask[i, len(tokens) - 1:] = True
    
    causal_mask = generate_square_subsequent_mask(T)
    return torch.tensor(src_ids), torch.tensor(tgt_ids), causal_mask, key_padding_mask

def collate_fn_ultrabigbrain(batch: list[list[int]], max_length: Optional[int] = MAX_LENGTH):
    return collate_fn_bigbrain(batch, max_length)

class UltraBigBrainBatchSampler(Sampler):

    def __init__(self, dataset: UltraBigBrainDataset, batch_size: int, max_length: Optional[int] = MAX_LENGTH, k: int = 1):
        self.batches = []
        self.batch_by_len = [[] for _ in range(max_length + 1)]
        rand_order = list(range(len(dataset)))
        random.shuffle(rand_order)
        for i in rand_order:
            el = dataset[i]
            rand_ind = random.randint(len(el), min(len(el) + k, max_length))
            self.batch_by_len[rand_ind].append(i)
            if len(self.batch_by_len[rand_ind]) == batch_size:
                self.batches.append(self.batch_by_len[rand_ind])
                self.batch_by_len[rand_ind] = []
        for length in range(1, max_length + 1):
            if len(self.batch_by_len[length]) > 0:
                self.batches.append(self.batch_by_len[length])

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)
