from enum import Enum

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from time import perf_counter
from tqdm.cli import tqdm
import numpy as np
import os

from .model import GPT2LikeModel
from .dataset import BrainDataset, BigBrainDataset, UltraBigBrainDataset, UltraBigBrainBatchSampler, UltraDuperBigBrainDataset
from .dataset import collate_fn_brain, collate_fn_bigbrain, collate_fn_ultrabigbrain

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model() -> torch.nn.Module:
    return GPT2LikeModel(tokenizer.vocab_size)

def get_dataloader(data_mode: DataMode, batch_size: int, k: int = 1):
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path="wikitext")
        return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn_brain, num_workers=4, pin_memory=True)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data_path="wikitext")
        return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn_bigbrain, num_workers=4, pin_memory=True)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN:
        dataset = UltraBigBrainDataset(data_path="wikitext")
        sampler = UltraBigBrainBatchSampler(dataset, batch_size, k=k)
        return DataLoader(dataset=dataset, batch_size=None, collate_fn=collate_fn_ultrabigbrain, sampler=sampler, num_workers=4, pin_memory=True)

    raise NotImplementedError

@torch.no_grad()
def run_epoch(data_mode: DataMode, batch_size: int = 64, warmup_batches: int = 5, k: int = 1) -> None:
    dataloader = get_dataloader(data_mode, batch_size, k)
    print("Dataloader is ready")
    device = "cuda"
    model = get_gpt2_model().to(device)
    print("Model is ready")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for i, batch in tqdm(enumerate(dataloader)):
        if i >= warmup_batches:
            break
        src, tgt, causal_mask, key_padding_mask = map(lambda x: x.to(device), batch)
        pred = model(src, causal_mask, key_padding_mask)
        loss = criterion(pred.flatten(0, 1), tgt.flatten(0, 1))
    
    torch.cuda.synchronize()
    batch_time = []
    for batch in tqdm(dataloader):
        start_time = perf_counter()
        src, tgt, causal_mask, key_padding_mask = map(lambda x: x.to(device), batch)
        pred = model(src, causal_mask, key_padding_mask)
        loss = criterion(pred.flatten(0, 1), tgt.flatten(0, 1))
        torch.cuda.synchronize()
        batch_time.append(perf_counter() - start_time)

    print("Batch proccessing time:")
    print("Min:", np.min(batch_time))
    print("Max:", np.max(batch_time))
    print("Mean:", np.mean(batch_time))
    print("Median:", np.median(batch_time))
