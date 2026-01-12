# probts/model/nn/arch/armd_layers.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Beta Schedules
# --------------------------------------------------------------------------

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# --------------------------------------------------------------------------
# Helper Modules
# --------------------------------------------------------------------------

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)

# --------------------------------------------------------------------------
# Main Backbone: Linear ARMD Network
# --------------------------------------------------------------------------

class LinearBackbone(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        timesteps, 
        w_grad=True,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(n_channel, n_channel)
        
        # Dynamic schedule based on prediction length (timesteps)
        self.betas = linear_beta_schedule(timesteps)
        self.betas_dev = cosine_beta_schedule(timesteps)
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_dev = 1. - self.betas_dev
        
        # Learnable parameters for weighting
        # We convert numpy/tensor to Parameter to ensure they move to device/fp16 correctly
        self.w = nn.Parameter(self.alphas_cumprod.float(), requires_grad=w_grad)
        self.w_dev = nn.Parameter(self.alphas_dev.float(), requires_grad=False)

    def forward(self, input_, t, training=True):
        # input_ shape: [Batch, Length, Channels/Features]
        
        # Deviation Noise logic
        noise = torch.randn_like(input_)
        if not training:
            noise = 0
            
        # Add deviation noise scaled by w_dev
        w_dev_t = extract(self.w_dev, t, input_.shape)
        input_ = input_ + w_dev_t * noise
        
        # Linear projection (permute to apply Linear on Length dimension)
        # Assuming input is [B, L, C], we want to mix L. 
        # original code: input_.permute(0,2,1) -> [B, C, L]
        # Linear is [L, L]. 
        x_tmp = self.linear(input_.permute(0,2,1)).permute(0,2,1)
        
        # Weighted combination (Eq 5 in ARMD paper)
        alpha = extract(self.w, t, input_.shape)
        
        # Avoid division by zero issues or instability if 1-alpha is tiny
        # The original code logic:
        output = (alpha * input_ + (1 - 2 * alpha) * x_tmp) / ((1 - alpha)**0.5 + 1e-8)
        
        return output