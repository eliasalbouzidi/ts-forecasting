# probts/model/nn/arch/armd_layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from probts.model.nn.arch.decomp import series_decomp

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
        alphas_cumprod=None,
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
        
        if alphas_cumprod is None:
            betas = linear_beta_schedule(timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = alphas_cumprod.detach().clone()

        # Learnable parameters for weighting W(t) initialized with alpha_bar
        self.w = nn.Parameter(alphas_cumprod.float(), requires_grad=w_grad)
        
        # Deviation weight eta_{0:t} (fixed schedule)
        self.register_buffer('w_dev', alphas_cumprod.float())

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


class DLinearBackbone(nn.Module):
    def __init__(
        self,
        n_feat,
        seq_len,
        timesteps,
        w_grad=True,
        alphas_cumprod=None,
        b_param=2.0,
        c_param=-1.0,
        d_param=0.5,
        dlinear_kernel_size=25,
        dlinear_individual=False,
        **kwargs
    ):
        super().__init__()
        self.n_feat = n_feat
        self.seq_len = seq_len
        self.individual = dlinear_individual

        self.b_param = b_param
        self.c_param = c_param
        self.d_param = d_param

        # DLinear-style decomposition: input = seasonal + trend
        self.decomposition = series_decomp(dlinear_kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.n_feat):
                self.linear_seasonal.append(nn.Linear(seq_len, seq_len))
                self.linear_trend.append(nn.Linear(seq_len, seq_len))
        else:
            self.linear_seasonal = nn.Linear(seq_len, seq_len)
            self.linear_trend = nn.Linear(seq_len, seq_len)

        if alphas_cumprod is None:
            betas = linear_beta_schedule(timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = alphas_cumprod.detach().clone()

        self.w = nn.Parameter(alphas_cumprod.float(), requires_grad=w_grad)
        self.register_buffer('w_dev', alphas_cumprod.float())

    def _dlinear_distance(self, input_):
        seasonal_init, trend_init = self.decomposition(input_)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros(
                seasonal_init.size(0), seasonal_init.size(1), self.seq_len,
                dtype=seasonal_init.dtype, device=seasonal_init.device
            )
            trend_out = torch.zeros(
                trend_init.size(0), trend_init.size(1), self.seq_len,
                dtype=trend_init.dtype, device=trend_init.device
            )
            for i in range(self.n_feat):
                seasonal_out[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_out[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_out = self.linear_seasonal(seasonal_init)
            trend_out = self.linear_trend(trend_init)

        return (seasonal_out + trend_out).permute(0, 2, 1)

    def forward(self, input_, t, training=True):
        noise = torch.randn_like(input_)
        if not training:
            noise = 0

        w_dev_t = extract(self.w_dev, t, input_.shape)
        input_ = input_ + w_dev_t * noise

        D = self._dlinear_distance(input_)

        W_t = extract(self.w, t, input_.shape)
        numerator = W_t * input_ + (1 - self.b_param * W_t) * D
        denominator = (1 + self.c_param * W_t).pow(self.d_param) + 1e-8
        output = numerator / denominator
        return output


class _MixerMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1, activation="gelu"):
        super().__init__()
        if activation.lower() == "relu":
            act = nn.ReLU()
        else:
            act = nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _TSMixerBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        n_feat,
        hidden_dim=128,
        temporal_mlp_ratio=2.0,
        channel_mlp_ratio=2.0,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()
        temporal_hidden = max(int(seq_len * temporal_mlp_ratio), int(hidden_dim))
        channel_hidden = max(int(n_feat * channel_mlp_ratio), int(hidden_dim))

        self.temporal_norm = nn.LayerNorm(seq_len)
        self.temporal_mlp = _MixerMLP(
            in_dim=seq_len,
            hidden_dim=temporal_hidden,
            dropout=dropout,
            activation=activation,
        )

        self.channel_norm = nn.LayerNorm(n_feat)
        self.channel_mlp = _MixerMLP(
            in_dim=n_feat,
            hidden_dim=channel_hidden,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        # Temporal mixing on each variable independently.
        t_in = x.permute(0, 2, 1)
        t_out = self.temporal_mlp(self.temporal_norm(t_in))
        x = x + t_out.permute(0, 2, 1)

        # Channel mixing on each timestep independently.
        c_out = self.channel_mlp(self.channel_norm(x))
        x = x + c_out
        return x


class TSMixerBackbone(nn.Module):
    def __init__(
        self,
        n_feat,
        seq_len,
        timesteps,
        w_grad=True,
        alphas_cumprod=None,
        b_param=2.0,
        c_param=-1.0,
        d_param=0.5,
        tsmixer_n_blocks=2,
        tsmixer_hidden_dim=128,
        tsmixer_dropout=0.1,
        tsmixer_temporal_mlp_ratio=2.0,
        tsmixer_channel_mlp_ratio=2.0,
        tsmixer_activation="gelu",
        **kwargs
    ):
        super().__init__()
        self.b_param = b_param
        self.c_param = c_param
        self.d_param = d_param

        self.blocks = nn.ModuleList(
            [
                _TSMixerBlock(
                    seq_len=seq_len,
                    n_feat=n_feat,
                    hidden_dim=tsmixer_hidden_dim,
                    temporal_mlp_ratio=tsmixer_temporal_mlp_ratio,
                    channel_mlp_ratio=tsmixer_channel_mlp_ratio,
                    dropout=tsmixer_dropout,
                    activation=tsmixer_activation,
                )
                for _ in range(int(tsmixer_n_blocks))
            ]
        )

        if alphas_cumprod is None:
            betas = linear_beta_schedule(timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            alphas_cumprod = alphas_cumprod.detach().clone()

        self.w = nn.Parameter(alphas_cumprod.float(), requires_grad=w_grad)
        self.register_buffer('w_dev', alphas_cumprod.float())

    def _tsmixer_distance(self, input_):
        x = input_
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, input_, t, training=True):
        noise = torch.randn_like(input_)
        if not training:
            noise = 0

        w_dev_t = extract(self.w_dev, t, input_.shape)
        input_ = input_ + w_dev_t * noise

        D = self._tsmixer_distance(input_)

        W_t = extract(self.w, t, input_.shape)
        numerator = W_t * input_ + (1 - self.b_param * W_t) * D
        denominator = (1 + self.c_param * W_t).pow(self.d_param) + 1e-8
        output = numerator / denominator
        return output
