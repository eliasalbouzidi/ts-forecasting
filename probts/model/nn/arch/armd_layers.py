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
    # Allows to perform element-wise multiplication between the scalar diffusion coefficients and the multi-dimensional time series data.
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# --------------------------------------------------------------------------
# Main Backbone: Linear ARMD Network (Devolution Network R)
# --------------------------------------------------------------------------

class LinearBackbone(nn.Module):
    def __init__(
        self,
        n_feat,
        seq_len,
        timesteps, 
        w_grad=True,
        # Hyperparameters for Eq (5) 
        b_param=2.0, 
        c_param=-1.0, 
        d_param=0.5,
        **kwargs
    ):
        super().__init__()
        self.linear = nn.Linear(seq_len, seq_len)
        
        # Hyperparameters
        self.b_param = b_param
        self.c_param = c_param
        self.d_param = d_param
        
        # Dynamic schedule based on prediction length (timesteps)
        self.betas = linear_beta_schedule(timesteps)
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.alphas_dev = 1. - self.betas
        # Deviation weight uses cumulative product alpha_bar 
        self.alphas_cumprod_dev = torch.cumprod(self.alphas_dev, dim=0)
        
        # Learnable parameters for weighting W(t) initialized with alpha_bar
        self.w = nn.Parameter(self.alphas_cumprod.float(), requires_grad=w_grad)
        
        # Deviation weight eta_{0:t}
        self.w_dev = nn.Parameter(self.alphas_cumprod.float(), requires_grad=False)

    def forward(self, input_, t, training=True):
        # input_ shape: [Batch, Length, Channels]
        
        # Deviation Noise logic 
        noise = torch.randn_like(input_)
        if not training:
            noise = 0
            
        # Add deviation noise scaled by w_dev (alpha_bar)
        # X_input = X + eta_{0:t} *epsilon
        w_dev_t = extract(self.w_dev, t, input_.shape)
        input_ = input_ + w_dev_t * noise
        
        # Linear projection to estimate Distance D 
        # Permute to [B, C, L] to apply linear on L
        D = self.linear(input_.permute(0,2,1)).permute(0,2,1)
        
        # Weighted combination Eq (5) 
        # Formula: (W(t) * X + (1 - b*W(t)) * D) / (1 + c*W(t))^d
        
        W_t = extract(self.w, t, input_.shape)
        
        numerator = W_t * input_ + (1 - self.b_param * W_t) * D
        denominator = (1 + self.c_param * W_t).pow(self.d_param) + 1e-8
        
        output = numerator / denominator
        
        return output