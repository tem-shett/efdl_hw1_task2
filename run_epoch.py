from enum import Enum

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from time import perf_counter
from tqdm.cli import tqdm
import numpy as np

from .model import GPT2LikeModel
from .dataset import BrainDataset, BigBrainDataset, UltraBigBrainDataset, UltraDuperBigBrainDataset
from .dataset import collate_fn_brain

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4


def get_gpt2_model() -> torch.nn.Module:
    return GPT2LikeModel(tokenizer.vocab_size)

def get_dataloader(data_mode, batch_size):
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data_path="wikitext")
        return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn_brain, num_workers=4, pin_memory=True)
    raise NotImplementedError

@torch.no_grad()
def run_epoch(data_mode: DataMode, batch_size: int = 64, warmup_batches: int = 5) -> None:
    dataloader = get_dataloader(data_mode, batch_size)
    device = "cuda"
    model = get_gpt2_model().to(device)
    dataloader = get_dataloader(data_mode, batch_size=batch_size)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for i, batch in tqdm(enumerate(dataloader)):
        if i >= warmup_batches:
            break
        src, tgt, causal_mask, key_padding_mask = map(lambda x: x.to(device), batch)
        pred = model(src, causal_mask, key_padding_mask)
        loss = criterion(pred.flatten(), tgt.flatten())
    
    torch.cuda.synchronize()
    batch_time = []
    for batch in tqdm(dataloader):
        start_time = perf_counter()
        src, tgt, causal_mask, key_padding_mask = map(lambda x: x.to(device), batch)
        pred = model(src, causal_mask, key_padding_mask)
        loss = criterion(pred.flatten(), tgt.flatten())
        torch.cuda.synchronize()
        batch_time.append(perf_counter() - start_time)

    print("Batch proccessing time:")
    print("Min:", np.min(batch_time))
    print("Max:", np.max(batch_time))
    print("Mean:", np.mean(batch_time))
    print("Median:", np.median(batch_time))
