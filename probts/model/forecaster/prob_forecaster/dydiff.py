import torch
from torch import nn
from torch.nn import functional as F

from probts.model.forecaster import Forecaster


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] float in [0, 1]
        half_dim = self.dim // 2
        if half_dim == 0:
            return t.unsqueeze(-1)
        emb_scale = torch.log(torch.tensor(10000.0, device=t.device)) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, t.unsqueeze(-1)], dim=-1)
        return emb


def _build_mlp(input_dim, hidden_dim, num_layers, dropout, output_dim):
    layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(max(num_layers - 1, 0)):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class Dydiff(Forecaster):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        horizon_embed_dim: int = 32,
        dropout: float = 0.1,
        num_diffusion_steps: int = 100,
        beta_end: float = 0.1,
        sample_chunk_size: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.horizon_embed_dim = horizon_embed_dim
        self.dropout = dropout
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.beta_end = float(beta_end)
        self.sample_chunk_size = int(sample_chunk_size) if sample_chunk_size is not None else None

        self.horizon = int(self.max_prediction_length)

        self.encoder = nn.GRU(
            input_size=self.target_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.time_embed = SinusoidalTimeEmbedding(self.time_embed_dim)
        self.horizon_embed = SinusoidalTimeEmbedding(self.horizon_embed_dim)

        self.dynamics = _build_mlp(
            input_dim=self.hidden_dim + self.horizon_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.target_dim,
        )

        denoise_in_dim = (
            self.target_dim
            + self.target_dim
            + self.hidden_dim
            + self.time_embed_dim
            + self.horizon_embed_dim
        )
        self.denoiser = _build_mlp(
            input_dim=denoise_in_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.target_dim,
        )

        self._init_diffusion_buffers()

    def _init_diffusion_buffers(self):
        betas = torch.linspace(1e-4, self.beta_end, self.num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas", torch.sqrt(1.0 / alphas)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def _extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def _normalize_time(self, t, max_val):
        denom = max(max_val - 1, 1)
        return t / denom

    def _encode_context(self, past_target):
        _, hidden = self.encoder(past_target)
        return hidden[-1]

    def _build_prior(self, context, horizon_len):
        device = context.device
        h_idx = torch.arange(horizon_len, device=device).float()
        h_norm = self._normalize_time(h_idx, horizon_len)
        h_emb = self.horizon_embed(h_norm)
        h_emb = h_emb.unsqueeze(0).expand(context.shape[0], -1, -1)
        context_rep = context.unsqueeze(1).expand(-1, horizon_len, -1)
        prior_inp = torch.cat([context_rep, h_emb], dim=-1)
        prior = self.dynamics(prior_inp.reshape(-1, prior_inp.shape[-1]))
        return prior.view(context.shape[0], horizon_len, self.target_dim), h_emb

    def _predict_eps(self, x_t, t_idx, context_rep, h_emb, prior):
        t_norm = self._normalize_time(t_idx.float(), self.num_diffusion_steps)
        t_emb = self.time_embed(t_norm)
        inp = torch.cat([x_t, prior, context_rep, t_emb, h_emb], dim=-1)
        return self.denoiser(inp)

    def loss(self, batch_data):
        past_target = batch_data.past_target_cdf[:, -self.max_context_length :, :]
        future_target = batch_data.future_target_cdf
        horizon_len = future_target.shape[1]

        context = self._encode_context(past_target)
        prior, h_emb = self._build_prior(context, horizon_len)

        bsz, horizon, _ = future_target.shape
        t_idx = torch.randint(
            0, self.num_diffusion_steps, (bsz, horizon), device=future_target.device
        )

        eps = torch.randn_like(future_target)
        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod, t_idx.reshape(-1), future_target.reshape(-1, self.target_dim)
        ).view(bsz, horizon, 1)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t_idx.reshape(-1),
            future_target.reshape(-1, self.target_dim),
        ).view(bsz, horizon, 1)

        x_t = sqrt_alpha_bar * future_target + sqrt_one_minus * eps

        x_t_flat = x_t.reshape(-1, self.target_dim)
        prior_flat = prior.reshape(-1, self.target_dim)
        context_rep = context.unsqueeze(1).expand(-1, horizon, -1).reshape(-1, self.hidden_dim)
        h_emb_flat = h_emb.reshape(-1, self.horizon_embed_dim)
        t_idx_flat = t_idx.reshape(-1)

        eps_hat = self._predict_eps(x_t_flat, t_idx_flat, context_rep, h_emb_flat, prior_flat)
        loss = F.mse_loss(eps_hat, eps.reshape(-1, self.target_dim), reduction="none")
        loss = loss.view(bsz, horizon, self.target_dim).mean(dim=-1)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        past_target = batch_data.past_target_cdf[:, -self.max_context_length :, :]
        if num_samples is None:
            num_samples = 1

        context = self._encode_context(past_target)
        horizon_len = int(self.max_prediction_length)
        prior, h_emb = self._build_prior(context, horizon_len)

        chunk_size = self.sample_chunk_size or num_samples
        chunk_size = max(1, min(chunk_size, num_samples))
        samples = []
        remaining = num_samples

        while remaining > 0:
            curr = min(chunk_size, remaining)
            remaining -= curr

            context_rep = context.repeat_interleave(curr, dim=0)
            prior_rep = prior.repeat_interleave(curr, dim=0)
            h_emb_rep = h_emb.repeat_interleave(curr, dim=0)

            bsz = context_rep.shape[0]
            x_t = torch.randn(bsz, horizon_len, self.target_dim, device=context_rep.device)

            for step in reversed(range(self.num_diffusion_steps)):
                t_idx = torch.full((bsz, horizon_len), step, device=context_rep.device, dtype=torch.long)
                x_t_flat = x_t.reshape(-1, self.target_dim)
                prior_flat = prior_rep.reshape(-1, self.target_dim)
                context_flat = context_rep.unsqueeze(1).expand(-1, horizon_len, -1).reshape(-1, self.hidden_dim)
                h_emb_flat = h_emb_rep.reshape(-1, self.horizon_embed_dim)
                t_idx_flat = t_idx.reshape(-1)

                eps_hat = self._predict_eps(x_t_flat, t_idx_flat, context_flat, h_emb_flat, prior_flat)

                alpha_t = self._extract(self.alphas, t_idx_flat, x_t_flat)
                sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t_idx_flat, x_t_flat)
                alpha_bar = self._extract(self.alphas_cumprod, t_idx_flat, x_t_flat)

                mean = (
                    sqrt_recip_alpha
                    * (x_t_flat - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar) * eps_hat)
                )

                if step > 0:
                    var = self._extract(self.posterior_variance, t_idx_flat, x_t_flat)
                    noise = torch.randn_like(x_t_flat)
                    x_t_flat = mean + torch.sqrt(var) * noise
                else:
                    x_t_flat = mean

                x_t = x_t_flat.view(bsz, horizon_len, self.target_dim)

            samples.append(x_t.view(-1, curr, horizon_len, self.target_dim))

        return torch.cat(samples, dim=1)
