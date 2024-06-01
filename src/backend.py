import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from model import ModelArgs, Transformer
from utlis import load_model, model_forward, prefill
class LMBackend:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0") -> None:
        
        self.dtype = dtype
        self.device = device
        self.model_forward = model_forward
        self.prefill = prefill
    
    def load_model(self, checkpoints: str):

        self.model: Transformer = load_model(checkpoint_path=checkpoints, device=self.device, precision=self.dtype)
    

    def compile(self, encode=False):
        import torch._dynamo.config
        import torch._inductor.config

        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
        self.model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)
        if encode:
             self.prefill = torch.compile(prefill, mode="reduce-overhead", fullgraph=True)
        
        
             

    @torch.inference_mode()
    @torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    def inference(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
            return self.model_forward(
                 model=self.model, 
                 x=input_ids,
                 input_pos=position_ids,
                 cache_pos=storage_ids,
                 attention_mask=attention_mask)
    
    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor, storage_ids: torch.LongTensor, attention_mask: torch.Tensor):
            return self.prefill(
                 model=self.model, 
                 x=input_ids,
                 input_pos=position_ids,
                 cache_pos=storage_ids,
                 attention_mask=attention_mask)
    



