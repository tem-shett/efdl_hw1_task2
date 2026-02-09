from functools import lru_cache
from typing import Optional

import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer
import random
from enum import Enum

from .transformer import generate_square_subsequent_mask


MAX_LENGTH = 640

@lru_cache()
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

class UltraBigBrainDataset(Dataset):
    pass

class Packing(Enum):
    Basic = 1
    FFD = 2
    OBFD = 3

class SegmentTree:
    def __init__(self, a):
        self.n = len(a)
        self.tree = [0] * (4 * self.n)
        self.build(0, 0, self.n, a)
    
    def build(self, i, l, r, a):
        if l + 1 == r:
            self.tree[i] = a[l]
            return
        m = (l + r) // 2
        self.build(i * 2 + 1, l, m, a)
        self.build(i * 2 + 2, m, r, a)
        self.tree[i] = max(self.tree[i * 2 + 1], self.tree[i * 2 + 2])

    def upd(self, pos, qx):
        self._upd(0, 0, self.n, pos, qx)
    
    def _upd(self, i, l, r, pos, qx):
        if l + 1 == r:
            self.tree[i] += qx
            return
        m = (l + r) // 2
        if pos < m:
            self._upd(i * 2 + 1, l, m, pos, qx)
        else:
            self._upd(i * 2 + 2, m, r, pos, qx)
        self.tree[i] = max(self.tree[i * 2 + 1], self.tree[i * 2 + 2])
    
    def find_first_more(self, val):
        res = -1

        def search(i, l, r):
            if r <= val:
                return
            if self.tree[i] == 0:
                return
            if res != -1:
                return
            if l + 1 == r:
                res = l
                return
            m = (l + r) // 2
            search(i * 2 + 1, l, m)
            search(i * 2 + 2, m, r)
        
        assert res != -1
        return res

class UltraDuperBigBrainDataset(WikiTextDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH, packing: Packing = Packing.Basic):
        super().__init__(data_path, max_length)
        self.input_ids_base = self.input_ids
        self.input_ids = []
        self.segment_ids = []

        if packing == Packing.Basic:
            cur_input_ids = []
            cur_seg_ids = []
            for i in range(len(self.input_ids_base)):
                if len(cur_input_ids) + len(self.input_ids_base[i]) > max_length:
                    self.input_ids.append(cur_input_ids)
                    self.segment_ids.append(cur_seg_ids)
                    cur_input_ids = []
                    cur_seg_ids = []
                cur_input_ids += self.input_ids_base[i]
                cur_seg_ids += [i] * len(self.input_ids_base[i])
            if cur_input_ids:
                self.input_ids.append(cur_input_ids)
                self.segment_ids.append(cur_seg_ids)
        
        elif packing == Packing.FFD:
            inds_by_length = [[] for _ in range(max_length + 1)]
            for i in range(len(self.input_ids_base)):
                inds_by_length[len(self.input_ids_base[i])].append(i)
            
            bins = []
            bin_smlen = []
            for length in range(max_length, 0, -1):
                for i in inds_by_length[length]:
                    flag = False
                    for j in range(len(bins)):
                        if bin_smlen[j] + length <= max_length:
                            bins[j].append(i)
                            bin_smlen[j] += length
                            flag = True
                            break
                    if not flag:
                        bins.append([i])
                        bin_smlen.append(length)
            for inds in bins:
                self.input_ids.append([])
                self.segment_ids.append([])
                for i in inds:
                    self.input_ids[-1] += self.input_ids_base[i]
                    self.segment_ids[-1] += [i] * len(self.input_ids_base[i])
        
        elif packing == Packing.OBFD:
            bins = [[] for _ in range(self.input_ids_base)]

            bins_by_left_length = [[] for _ in range(max_length + 1)]
            for i in range(len(self.input_ids_base)):
                bins_by_left_length[max_length].append(i)
            
            st_a = [0] * (max_length + 1)
            st_a[max_length] = len(self.input_ids_base)
            st = SegmentTree(st_a)

            inds_by_length = [[] for _ in range(max_length + 1)]
            for i in range(len(self.input_ids_base)):
                inds_by_length[len(self.input_ids_base[i])].append(i)
            
            for length in range(max_length, 0, -1):
                for i in inds_by_length[length]:
                    left_length = st.find_first_more(length)
                    ind_bin = bins_by_left_length[left_length].pop()
                    st.upd(left_length, -1)
                    bins[ind_bin].append(i)
                    bins_by_left_length[left_length - length].append(ind_bin)
                    st.upd(left_length - length, 1)
            
            for inds in bins:
                if not inds:
                    continue
                self.input_ids.append([])
                self.segment_ids.append([])
                for i in inds:
                    self.input_ids[-1] += self.input_ids_base[i]
                    self.segment_ids[-1] += [i] * len(self.input_ids_base[i])
            


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx]


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
    return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), causal_mask, key_padding_mask


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
    return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), causal_mask, key_padding_mask

def collate_fn_ultrabigbrain(batch: list[list[int]], max_length: Optional[int] = MAX_LENGTH):
    return collate_fn_bigbrain(batch, max_length)

def collate_fn_ultaduperbigbrain(batch: list[tuple[list[int], list[int]]], max_length: Optional[int] = MAX_LENGTH):
    B = len(batch)
    T = max(len(el[0]) for el in batch) - 1
    src_ids = []
    tgt_ids = []
    batch_seg_ids = []
    for tokens, segs in batch:
        pad_len = T + 1 - len(tokens)
        padded_tokens = tokens + [0] * pad_len
        padded_segs = segs + [0] * pad_len
        src_ids.append(padded_tokens[:-1])
        tgt_ids.append(padded_tokens[1:])
        batch_seg_ids.append(padded_segs[:-1])
    
    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
    seg_tensor = torch.tensor(batch_seg_ids, dtype=torch.long)

    key_padding_mask = (src_tensor == 0)

    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    cross_seg_mask = (seg_tensor.unsqueeze(2) != seg_tensor.unsqueeze(1))
    combined_mask = causal_mask | cross_seg_mask
    final_mask = torch.zeros_like(combined_mask, dtype=torch.float)
    final_mask.masked_fill_(combined_mask, float('-inf'))

    return src_tensor, tgt_tensor, final_mask, key_padding_mask


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
        current_batch = []
        for length in range(1, max_length + 1):
            for ind in self.batch_by_len[length]:
                if (len(current_batch) > 0 and abs(len(dataset[current_batch[0]]) - len(dataset[ind])) > k) or len(current_batch) == batch_size:
                    self.batches.append(current_batch)
                    current_batch = []
                current_batch.append(ind)
        if current_batch:
            self.batches.append(current_batch)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)
