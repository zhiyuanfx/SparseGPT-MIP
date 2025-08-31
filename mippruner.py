import time

import torch
import torch.nn as nn
import transformers

from quant import *
from mask_solver import solve_mask

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class MIPPruner:
    
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        
        W = layer.weight.data.clone()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        elif not isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            raise NotImplementedError("Only Linear and Conv1D layers are supported currently.")
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.inps = None  # inps shape: [nsamples, input_dim]
        self.outs = None  # outs shape: [nsamples, output_dim]

    def add_batch(self, inp, out):
        inp = inp.reshape((-1, inp.shape[-1]))  # inp shape: [cur_nsample * seq_len, input_dim]
        out = out.reshape((-1, out.shape[-1]))  # out shape: [cur_nsample * seq_len, output_dim]
    
        self.inps = inp if self.inps is None else torch.cat((self.inps, inp), dim=0)
        self.outs = out if self.outs is None else torch.cat((self.outs, out), dim=0)

    def gurobi_mip_prune(self, sparsity, n, m, structure='unstructured', num_processes=None):
        
        time0 = time.time()
        W = self.layer.weight.data.clone()
        
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer') and not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        inps = self.inps
        outs = self.outs
        if self.layer.bias is None:
            debiased_outs = outs
        else:
            debiased_outs = outs - self.layer.bias.data.unsqueeze(0)
        W = solve_mask(inps, debiased_outs, W, sparsity, n, m, method='gurobi', structure=structure)

        if hasattr(self, 'quantizer'):
            W_q = W.clone()
            for i in range(W_q.shape[1]):
                W_q[:, i] = quantize(
                    W_q[:, i].unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
            W = W_q

        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        time1 = time.time()
        
        prune_loss = torch.nn.functional.mse_loss(outs, self.layer(inps)).item()
        print(f"time: {time1 - time0:.2f}s, prune loss: {prune_loss:.10f}")

    def free(self):
        self.inps = None
        self.outs = None
        torch.cuda.empty_cache()