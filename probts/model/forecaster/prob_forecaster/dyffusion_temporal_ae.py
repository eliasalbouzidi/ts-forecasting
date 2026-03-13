import torch
from torch import nn
from torch.nn import functional as F

from probts.model.forecaster import Forecaster


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
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


class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        mlp_in_dim = input_dim + time_embed_dim
        layers = [nn.Linear(mlp_in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


class PerVariableMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: [B, K, in_dim]
        bsz, target_dim, in_dim = x.shape
        x = x.reshape(bsz * target_dim, in_dim)
        y = self.net(x)
        return y.reshape(bsz, target_dim, -1)


class DyffusionTemporalAE(Forecaster):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
        num_diffusion_steps: int = None,
        sampling_steps: int = None,
        lookahead_loss: bool = True,
        condition_on_xt: bool = False,
        stochastic_interpolator: bool = True,
        latent_dim_per_var: int = 8,
        temporal_encoder_hidden_dim: int = 128,
        ae_recon_weight: float = 1.0,
        ae_pred_weight: float = 1.0,
        latent_align_weight: float = 1.0,
        dyffusion_weight: float = 1.0,
        endpoint_ae_weight: float = 1.0,
        horizon_pred_weight: float = 1.0,
        encoder_context_length: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if latent_dim_per_var <= 1:
            raise ValueError("latent_dim_per_var must be > 1 to avoid 1-to-1 latent bottleneck")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.lookahead_loss = lookahead_loss
        self.condition_on_xt = condition_on_xt
        self.stochastic_interpolator = stochastic_interpolator

        self.latent_dim_per_var = int(latent_dim_per_var)
        self.temporal_encoder_hidden_dim = int(temporal_encoder_hidden_dim)
        self.ae_recon_weight = float(ae_recon_weight)
        self.ae_pred_weight = float(ae_pred_weight)
        self.latent_align_weight = float(latent_align_weight)
        self.dyffusion_weight = float(dyffusion_weight)
        self.endpoint_ae_weight = float(endpoint_ae_weight)
        self.horizon_pred_weight = float(horizon_pred_weight)

        self.horizon = int(self.prediction_length)
        self.num_diffusion_steps = int(num_diffusion_steps or self.horizon)
        self.sampling_steps = int(sampling_steps or self.num_diffusion_steps)

        self.encoder_context_length = int(encoder_context_length or self.context_length)
        self.latent_dim = self.target_dim * self.latent_dim_per_var

        # Temporal autoencoder over a history window, encoded per variable.
        self.temporal_encoder = PerVariableMLP(
            in_dim=self.encoder_context_length,
            hidden_dim=self.temporal_encoder_hidden_dim,
            out_dim=self.latent_dim_per_var,
            dropout=self.dropout,
        )
        self.temporal_decoder = PerVariableMLP(
            in_dim=self.latent_dim_per_var,
            hidden_dim=self.temporal_encoder_hidden_dim,
            out_dim=self.encoder_context_length,
            dropout=self.dropout,
        )

        # Point encoder/decoder to connect latent states to endpoint values.
        self.point_encoder = PerVariableMLP(
            in_dim=1,
            hidden_dim=max(self.temporal_encoder_hidden_dim // 2, 8),
            out_dim=self.latent_dim_per_var,
            dropout=self.dropout,
        )
        self.point_decoder = PerVariableMLP(
            in_dim=self.latent_dim_per_var,
            hidden_dim=max(self.temporal_encoder_hidden_dim // 2, 8),
            out_dim=1,
            dropout=self.dropout,
        )

        interp_input_dim = self.latent_dim * 2
        self.interpolator = TimeConditionedMLP(
            input_dim=interp_input_dim,
            time_embed_dim=self.time_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.latent_dim,
        )

        forecaster_input_dim = self.latent_dim + (self.latent_dim if self.condition_on_xt else 0)
        self.forecaster = TimeConditionedMLP(
            input_dim=forecaster_input_dim,
            time_embed_dim=self.time_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.latent_dim,
        )

    def _normalize_time(self, t):
        denom = max(self.horizon - 1, 1)
        return t / denom

    def _build_schedule(self, num_steps, device):
        if num_steps <= 1:
            return torch.zeros(1, device=device)
        return torch.linspace(0, self.horizon - 1, steps=num_steps, device=device)

    def _flatten_latent(self, z):
        return z.reshape(z.shape[0], -1)

    def _unflatten_latent(self, z_flat):
        return z_flat.view(z_flat.shape[0], self.target_dim, self.latent_dim_per_var)

    def _encode_temporal_window(self, past_window):
        # past_window: [B, L, K] -> [B, K, latent_dim_per_var]
        if past_window.shape[1] != self.encoder_context_length:
            raise ValueError(
                f"Expected context window length {self.encoder_context_length}, got {past_window.shape[1]}"
            )
        per_var_series = past_window.permute(0, 2, 1)
        return self.temporal_encoder(per_var_series)

    def _decode_temporal_window(self, z):
        # z: [B, K, latent_dim_per_var] -> [B, L, K]
        per_var_series = self.temporal_decoder(z)
        return per_var_series.permute(0, 2, 1)

    def _encode_point(self, x):
        # x: [B, K] -> [B, K, latent_dim_per_var]
        return self.point_encoder(x.unsqueeze(-1))

    def _decode_point(self, z):
        # z: [B, K, latent_dim_per_var] -> [B, K]
        return self.point_decoder(z).squeeze(-1)

    def _interpolate(self, z_t, z_th, t_idx):
        inp = torch.cat([z_t, z_th], dim=-1)
        t_norm = self._normalize_time(t_idx)
        pred = self.interpolator(inp, t_norm)
        if t_idx.ndim == 0:
            return z_t if t_idx.item() == 0 else pred
        mask = (t_idx == 0).unsqueeze(-1)
        return torch.where(mask, z_t, pred)

    def _forecast(self, s_n, t_idx, z_t=None):
        if self.condition_on_xt and z_t is not None:
            inp = torch.cat([s_n, z_t], dim=-1)
        else:
            inp = s_n
        t_norm = self._normalize_time(t_idx)
        return self.forecaster(inp, t_norm)

    def _dyffusion_loss(self, z_t, z_th, schedule):
        bsz = z_t.shape[0]
        device = z_t.device
        n_idx = torch.randint(0, schedule.shape[0], (bsz,), device=device)
        i_n = schedule[n_idx]

        if self.stochastic_interpolator:
            self.interpolator.train()
        else:
            self.interpolator.eval()

        s_n = torch.where(i_n.unsqueeze(-1) == 0, z_t, self._interpolate(z_t, z_th, i_n))
        z_th_pred = self._forecast(s_n, i_n, z_t=z_t)
        loss = F.mse_loss(z_th_pred, z_th, reduction="none").mean(dim=-1)

        if self.lookahead_loss and schedule.shape[0] > 1:
            valid = (n_idx + 1 < schedule.shape[0])
            if valid.any():
                n_next = torch.clamp(n_idx + 1, max=schedule.shape[0] - 1)
                i_next = schedule[n_next]
                s_next = self._interpolate(z_t, z_th_pred, i_next)
                z_th_pred_next = self._forecast(s_next, i_next, z_t=z_t)
                loss_next = F.mse_loss(z_th_pred_next, z_th, reduction="none").mean(dim=-1)
                loss = torch.where(valid, 0.5 * (loss + loss_next), loss)

        return loss.mean()

    def _decode_forecast_trajectory(self, z_t, z_th_pred, device):
        z_th_pred_k = self._unflatten_latent(z_th_pred)
        x_th_pred = self._decode_point(z_th_pred_k)

        if self.horizon == 1:
            return x_th_pred.unsqueeze(1), x_th_pred

        j_idx = torch.arange(1, self.horizon, device=device).float()
        j_rep = j_idx.unsqueeze(0).repeat(z_t.shape[0], 1).reshape(-1)
        z_t_rep = z_t.repeat_interleave(self.horizon - 1, dim=0)
        z_th_rep = z_th_pred.repeat_interleave(self.horizon - 1, dim=0)
        z_tj = self._interpolate(z_t_rep, z_th_rep, j_rep).view(
            z_t.shape[0], self.horizon - 1, self.latent_dim
        )
        z_tj_k = z_tj.view(-1, self.target_dim, self.latent_dim_per_var)
        x_tj = self._decode_point(z_tj_k).view(z_t.shape[0], self.horizon - 1, self.target_dim)
        forecasts = torch.cat([x_tj, x_th_pred.unsqueeze(1)], dim=1)
        return forecasts, x_th_pred

    def loss(self, batch_data):
        past_window = batch_data.past_target_cdf[:, -self.encoder_context_length :, :]
        x_th_true = batch_data.future_target_cdf[:, -1, :]

        z_t_k = self._encode_temporal_window(past_window)
        z_t = self._flatten_latent(z_t_k)
        z_th_true_k = self._encode_point(x_th_true)
        z_th_true = self._flatten_latent(z_th_true_k)

        recon_window = self._decode_temporal_window(z_t_k)
        recon_loss = F.mse_loss(recon_window, past_window)

        # Ensure point encoder/decoder does not collapse to a near-constant mapping.
        x_th_recon_true = self._decode_point(z_th_true_k)
        endpoint_ae_loss = F.mse_loss(x_th_recon_true, x_th_true)

        schedule = self._build_schedule(self.num_diffusion_steps, z_t.device)
        dyffusion_loss = self._dyffusion_loss(z_t, z_th_true, schedule)

        # Supervise endpoint prediction at i=h-1 to match inference target x_{t+h}.
        i_h = torch.full((z_t.shape[0],), float(self.horizon - 1), device=z_t.device)
        z_th_pred = self._forecast(z_t, i_h, z_t=z_t)
        horizon_pred, x_th_pred = self._decode_forecast_trajectory(z_t, z_th_pred, z_t.device)
        future_true = batch_data.future_target_cdf[:, : self.horizon, :]

        pred_loss = F.mse_loss(x_th_pred, x_th_true)
        latent_align_loss = F.mse_loss(z_th_pred, z_th_true)
        horizon_pred_loss = F.mse_loss(horizon_pred, future_true)

        total_loss = (
            self.ae_recon_weight * recon_loss
            + self.endpoint_ae_weight * endpoint_ae_loss
            + self.dyffusion_weight * dyffusion_loss
            + self.ae_pred_weight * pred_loss
            + self.latent_align_weight * latent_align_loss
            + self.horizon_pred_weight * horizon_pred_loss
        )
        return total_loss

    def forecast(self, batch_data, num_samples=None):
        past_window = batch_data.past_target_cdf[:, -self.encoder_context_length :, :]
        z_t_k = self._encode_temporal_window(past_window)
        z_t = self._flatten_latent(z_t_k)

        if num_samples is None:
            num_samples = 1
        if num_samples > 1:
            z_t = z_t.repeat_interleave(num_samples, dim=0)

        device = z_t.device
        schedule = self._build_schedule(self.sampling_steps, device)

        if self.stochastic_interpolator:
            self.interpolator.train()
        else:
            self.interpolator.eval()
        self.forecaster.eval()

        s_n = z_t
        z_th_pred = None
        for n in range(schedule.shape[0]):
            i_n = schedule[n].expand(z_t.shape[0])
            z_th_pred = self._forecast(s_n, i_n, z_t=z_t)
            if n + 1 >= schedule.shape[0]:
                break
            i_next = schedule[n + 1].expand(z_t.shape[0])
            interp_next = self._interpolate(z_t, z_th_pred, i_next)
            interp_curr = self._interpolate(z_t, z_th_pred, i_n)
            s_n = interp_next - interp_curr + s_n

        forecasts, _ = self._decode_forecast_trajectory(z_t, z_th_pred, device)

        forecasts = forecasts.view(-1, num_samples, self.horizon, self.target_dim)
        return forecasts