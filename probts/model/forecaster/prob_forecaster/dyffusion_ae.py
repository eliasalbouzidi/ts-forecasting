import torch
from torch import nn
from torch.nn import functional as F

from probts.model.forecaster import Forecaster
from probts.model.forecaster.prob_forecaster.dyffusion import (
    SinusoidalTimeEmbedding,
    TimeConditionedMLP,
)


class SimpleEncoder(nn.Module):
    """Simple encoder that reduces dimension from target_dim to latent_dim"""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleDecoder(nn.Module):
    """Simple decoder that expands dimension from latent_dim to target_dim"""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DyffusionAE(Forecaster):
    """Dyffusion with AutoEncoder: Encoder -> Dyffusion -> Decoder"""

    def __init__(
        self,
        latent_dim: int = 32,
        encoder_hidden_dim: int = 128,
        decoder_hidden_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        time_embed_dim: int = 64,
        dropout: float = 0.1,
        num_diffusion_steps: int = None,
        sampling_steps: int = None,
        training_stage: str = "forecaster",
        lookahead_loss: bool = True,
        condition_on_xt: bool = False,
        stochastic_interpolator: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.lookahead_loss = lookahead_loss
        self.training_stage = training_stage
        self.condition_on_xt = condition_on_xt
        self.stochastic_interpolator = stochastic_interpolator

        self.horizon = int(self.prediction_length)
        self.num_diffusion_steps = int(num_diffusion_steps or self.horizon)
        self.sampling_steps = int(sampling_steps or self.num_diffusion_steps)

        # Encoder: target_dim -> latent_dim
        self.encoder = SimpleEncoder(
            input_dim=self.target_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.encoder_hidden_dim,
        )

        # Decoder: latent_dim -> target_dim
        self.decoder = SimpleDecoder(
            latent_dim=self.latent_dim,
            output_dim=self.target_dim,
            hidden_dim=self.decoder_hidden_dim,
        )

        # Dyffusion operates on latent space
        interp_input_dim = self.latent_dim * 2
        self.interpolator = TimeConditionedMLP(
            input_dim=interp_input_dim,
            time_embed_dim=self.time_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.latent_dim,
        )

        forecaster_input_dim = self.latent_dim + (
            self.latent_dim if self.condition_on_xt else 0
        )
        self.forecaster = TimeConditionedMLP(
            input_dim=forecaster_input_dim,
            time_embed_dim=self.time_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.latent_dim,
        )

    def _set_requires_grad(self, module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def _normalize_time(self, t):
        denom = max(self.horizon - 1, 1)
        return t / denom

    def _build_schedule(self, num_steps, device):
        if num_steps <= 1:
            return torch.zeros(1, device=device)
        return torch.linspace(0, self.horizon - 1, steps=num_steps, device=device)

    def _encode(self, x):
        """Encode input from original space to latent space"""
        # x: [B, L, target_dim] or [B, target_dim]
        original_shape = x.shape
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        latent = self.encoder(x)
        if len(original_shape) == 3:
            latent = latent.reshape(original_shape[0], original_shape[1], self.latent_dim)
        return latent

    def _decode(self, z):
        """Decode from latent space back to original space"""
        # z: [B, L, latent_dim] or [B, latent_dim]
        original_shape = z.shape
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])
        x = self.decoder(z)
        if len(original_shape) == 3:
            x = x.reshape(original_shape[0], original_shape[1], self.target_dim)
        return x

    def _interpolate(self, z_t, z_th, t_idx):
        """Interpolate in latent space (analogous to Dyffusion._interpolate)"""
        inp = torch.cat([z_t, z_th], dim=-1)
        t_norm = self._normalize_time(t_idx)
        pred = self.interpolator(inp, t_norm)
        if t_idx.ndim == 0:
            return z_t if t_idx.item() == 0 else pred
        mask = (t_idx == 0).unsqueeze(-1)
        return torch.where(mask, z_t, pred)

    def _forecast(self, s_n, t_idx, z_t=None):
        """Forecast in latent space (analogous to Dyffusion._forecast)"""
        if self.condition_on_xt and z_t is not None:
            inp = torch.cat([s_n, z_t], dim=-1)
        else:
            inp = s_n
        t_norm = self._normalize_time(t_idx)
        return self.forecaster(inp, t_norm)

    def _stage1_interpolator_loss(self, z_t, z_th, z_ti, i_idx):
        """Loss for interpolator training in latent space"""
        pred_ti = self._interpolate(z_t, z_th, i_idx)
        return F.mse_loss(pred_ti, z_ti, reduction="none").mean(dim=-1)

    def _stage2_forecaster_loss(self, z_t, z_th, schedule):
        """Loss for forecaster training in latent space"""
        bsz = z_t.shape[0]
        device = z_t.device
        n_idx = torch.randint(0, schedule.shape[0], (bsz,), device=device)
        i_n = schedule[n_idx]

        if self.stochastic_interpolator:
            self.interpolator.train()
        else:
            self.interpolator.eval()

        s_n = torch.where(
            i_n.unsqueeze(-1) == 0, z_t, self._interpolate(z_t, z_th, i_n)
        )
        z_th_pred = self._forecast(s_n, i_n, z_t=z_t)
        loss = F.mse_loss(z_th_pred, z_th, reduction="none").mean(dim=-1)

        return loss

    def training_step(self, batch, batch_idx):
        x, y, x_mark, y_mark = batch

        # Encode to latent space
        z_t_full = self._encode(x)  # [B, context_length, latent_dim]
        z_th_full = self._encode(y)  # [B, prediction_length, latent_dim]
        
        # Use last timestep for forecasting in diffusion model
        z_t = z_t_full[:, -1, :]  # [B, latent_dim]
        z_th = z_th_full[:, -1, :]  # [B, latent_dim]

        schedule = self._build_schedule(
            self.num_diffusion_steps if self.training_stage == "interpolator" else 100,
            z_t.device,
        )

        if self.training_stage == "interpolator":
            # Use linear interpolation as ground truth, not the network's own output
            z_ti_list = [z_t]
            for i_idx in range(1, schedule.shape[0]):
                # Linear interpolation: z_ti = (1 - alpha) * z_t + alpha * z_th
                alpha = float(i_idx) / float(schedule.shape[0] - 1)
                z_ti = (1 - alpha) * z_t + alpha * z_th
                z_ti_list.append(z_ti)
            z_ti = torch.stack(z_ti_list, dim=1)  # [B, num_interp_steps, latent_dim]

            i_idx = torch.arange(
                1, z_ti.shape[1], dtype=z_t.dtype, device=z_t.device
            )
            loss = self._stage1_interpolator_loss(z_t, z_th, z_ti[:, 1:, :], i_idx[: z_ti.shape[1] - 1])

        else:
            loss = self._stage2_forecaster_loss(z_t, z_th, schedule)

        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, prog_bar=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        x, y, x_mark, y_mark = batch

        # Encode to latent space
        z_t_full = self._encode(x)  # [B, context_length, latent_dim]
        z_th_full = self._encode(y)  # [B, prediction_length, latent_dim]
        
        # Use last timestep for forecasting in diffusion model
        z_t = z_t_full[:, -1, :]  # [B, latent_dim]
        z_th = z_th_full[:, -1, :]  # [B, latent_dim]

        schedule = self._build_schedule(self.num_diffusion_steps, z_t.device)

        if self.training_stage == "interpolator":
            # Use linear interpolation as ground truth
            z_ti_list = [z_t]
            for i_idx in range(1, schedule.shape[0]):
                # Linear interpolation: z_ti = (1 - alpha) * z_t + alpha * z_th
                alpha = float(i_idx) / float(schedule.shape[0] - 1)
                z_ti = (1 - alpha) * z_t + alpha * z_th
                z_ti_list.append(z_ti)
            z_ti = torch.stack(z_ti_list, dim=1)

            i_idx = torch.arange(
                1, z_ti.shape[1], dtype=z_t.dtype, device=z_t.device
            )
            loss = self._stage1_interpolator_loss(z_t, z_th, z_ti[:, 1:, :], i_idx[: z_ti.shape[1] - 1])
        else:
            loss = self._stage2_forecaster_loss(z_t, z_th, schedule)

        self.log("val_loss", loss.mean(), on_step=False, on_epoch=True, prog_bar=True)

    def sampling_step(self, x, y=None, num_sample=100, deterministic=False):
        """Inference: sample forecasts in latent space then decode"""
        x = self._encode(x)  # [B, context_length, latent_dim]

        b, _, _ = x.shape
        device = x.device

        schedule = self._build_schedule(self.sampling_steps, device)
        samples = []

        for _ in range(num_sample):
            s_t = x[:, -1:, :]  # Start from last context
            z_list = [s_t]

            for n_idx in range(len(schedule) - 1):
                i_n = int(schedule[n_idx + 1].item())
                s_n = self._interpolate(x[:, -1:, :], z_list[-1], torch.tensor(i_n, device=device))
                z_th_pred = self._forecast(s_n, torch.tensor(i_n, device=device), z_t=x[:, -1:, :])
                z_list.append(z_th_pred)

            z_pred = torch.cat(z_list[1:], dim=1)  # [B, prediction_length, latent_dim]
            samples.append(z_pred)

        # Decode back to original space
        z_samples = torch.stack(samples, dim=0)  # [num_sample, B, prediction_length, latent_dim]
        x_samples = self._decode(z_samples)  # [num_sample, B, prediction_length, target_dim]

        return x_samples

    def loss(self, batch_data):
        """Compute loss in latent space"""
        # Encode batch data to latent space
        z_t_full = self._encode(batch_data.past_target_cdf)  # [B, context_length, latent_dim]
        z_th_full = self._encode(batch_data.future_target_cdf)  # [B, prediction_length, latent_dim]
        
        # Use last timestep of context for forecasting
        z_t = z_t_full[:, -1, :]  # [B, latent_dim]
        z_th = z_th_full[:, -1, :]  # [B, latent_dim]

        schedule = self._build_schedule(self.num_diffusion_steps, z_t.device)

        stage = self.training_stage.lower()
        if stage == "interpolator":
            self._set_requires_grad(self.interpolator, True)
            self._set_requires_grad(self.forecaster, False)
            self.interpolator.train()
            self.forecaster.eval()

            if self.horizon < 2:
                return torch.zeros((), device=z_t.device)

            # Sample random interpolation indices for this batch
            i_idx = torch.randint(1, self.horizon, (z_t.shape[0],), device=z_t.device).float()
            
            # Compute true interpolation targets using linear interpolation
            alpha = i_idx / float(self.horizon)  # Normalize to [0, 1]
            alpha = alpha.unsqueeze(-1)  # [B, 1]
            z_ti_true = (1 - alpha) * z_t.unsqueeze(1) + alpha * z_th.unsqueeze(1)  # [B, 1, latent_dim]
            z_ti_true = z_ti_true.squeeze(1)  # [B, latent_dim]
            
            loss = self._stage1_interpolator_loss(z_t, z_th, z_ti_true, i_idx)
            return loss.mean()

        if stage == "forecaster":
            self._set_requires_grad(self.interpolator, False)
            self._set_requires_grad(self.forecaster, True)
            self.forecaster.train()
            loss = self._stage2_forecaster_loss(z_t, z_th, schedule)
            return loss.mean()

        if stage == "joint":
            self._set_requires_grad(self.interpolator, True)
            self._set_requires_grad(self.forecaster, True)
            self.interpolator.train()
            self.forecaster.train()

            if self.horizon < 2:
                return self._stage2_forecaster_loss(z_t, z_th, schedule).mean()

            i_idx = torch.randint(1, self.horizon, (z_t.shape[0],), device=z_t.device).float()
            loss_interp = self._stage1_interpolator_loss(z_t, z_th, z_th, i_idx)
            loss_forecast = self._stage2_forecaster_loss(z_t, z_th, schedule)
            return (loss_interp + loss_forecast).mean()

        raise ValueError(f"Unknown training_stage: {self.training_stage}")

    def forecast(self, batch_data, num_samples=None):
        """Forecast in latent space then decode"""
        # Encode full context sequence for richness
        z_t_full = self._encode(batch_data.past_target_cdf)  # [B, context_length, latent_dim]
        # Use last timestep of context for forecasting
        z_t = z_t_full[:, -1, :]  # [B, latent_dim]
        
        if num_samples is None:
            num_samples = 1
        if num_samples > 1:
            z_t = z_t.repeat_interleave(num_samples, dim=0)

        device = z_t.device
        bsz = z_t.shape[0]
        
        self.forecaster.eval()
        self.interpolator.eval()
        
        # Step 1: Predict final timestep z_th_pred from z_t
        # Use multiple iterations to refine the prediction
        z_th_pred = z_t.clone()
        schedule = self._build_schedule(self.sampling_steps, device)
        
        for step_idx in range(len(schedule) - 1):
            t_curr = schedule[step_idx]
            t_next = schedule[step_idx + 1]
            
            # Predict next point
            z_next = self._forecast(z_th_pred, t_next.expand(bsz), z_t=z_t)
            z_th_pred = z_next

        # Step 2: Generate full trajectory by interpolating between z_t and z_th_pred
        # For each horizon h=1..H, generate z_(t+h)
        z_forecasts_list = []
        
        for h in range(1, self.horizon + 1):
            t_norm = float(h) / float(self.horizon)
            t_tensor = torch.tensor(h, device=device, dtype=torch.float32)
            
            # Interpolate between z_t and z_th_pred at time h
            z_h = self._interpolate(z_t, z_th_pred, t_tensor.expand(bsz))
            z_forecasts_list.append(z_h)
        
        z_forecasts = torch.stack(z_forecasts_list, dim=1)  # [B, horizon, latent_dim]
        
        # Step 3: Decode back to original space
        forecasts = self._decode(z_forecasts)  # [B, horizon, target_dim]
        
        if num_samples > 1:
            forecasts = forecasts.view(-1, num_samples, self.horizon, self.target_dim)
        else:
            forecasts = forecasts.unsqueeze(1)
        
        return forecasts
