import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
from model import Transformer
import argparse
from backend import LMBackend


parser = argparse.ArgumentParser(description='Your CLI description.')

parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--batch', type=int, help='batch size')
parser.add_argument('--maxlen', type=int, help='max len')
parser.add_argument('--declen', type=int, help='decode len')
parser.add_argument('--prefixlen', type=int, help='prefix len')
parser.add_argument('--device', type=str, help='device')
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
device = args.device
precision = torch.bfloat16
use_tp = False
max_seq_length = args.maxlen
max_batch_size = args.batch
prefix_len = args.prefixlen
declen = args.declen
warm_up = 10
T = 500

causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device='cuda:9'))

llm = LMBackend(dtype=precision, device=device)
llm.load_model(checkpoint_path)
if args.compile:
    llm.compile()

with torch.device(device):
        llm.model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)


prompt = torch.tensor([[    1, 15043, 29892,   590,  1024,   338]], device='cuda:9',
       dtype=torch.int32)
input_pos = torch.tensor([0, 1, 2, 3, 4, 5], device='cuda:9')
mask = causal_mask[:6]

dec =  torch.tensor([[518]], device='cuda:9', dtype=torch.int32)
dec_pos = torch.tensor([6], device='cuda:9', dtype=torch.int32)
cache_pos = torch.tensor([6], device='cuda:9', dtype=torch.int32)
dec_mask = causal_mask[6:7][None, None, :, :]

dec1 =  torch.tensor([[627]], device='cuda:9', dtype=torch.int32)
dec_pos1 = torch.tensor([7], device='cuda:9', dtype=torch.int32)
cache_pos1 = torch.tensor([7], device='cuda:9', dtype=torch.int32)
dec_mask1 = causal_mask[7:8][None, None, :, :]

with torch.inference_mode():
        logits = llm.encode(input_ids=prompt, position_ids=input_pos, storage_ids=None, attention_mask=None)
        print(logits)

        logits = llm.inference(input_ids=dec, position_ids=dec_pos, storage_ids=cache_pos, attention_mask=dec_mask)
        # logits = model_forward(model=model,x=dec,input_pos=dec_pos, cache_pos=cache_pos, attention_mask=dec_mask)
        print(logits)

        logits = llm.inference(input_ids=dec1, position_ids=dec_pos1, storage_ids=cache_pos1, attention_mask=dec_mask1)
        # logits = model_forward(model=model,x=dec1,input_pos=dec_pos1, cache_pos=cache_pos1, attention_mask=dec_mask1)
        print(logits)
        











