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


class Dyffusion(Forecaster):
    def __init__(
        self,
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

        interp_input_dim = self.target_dim * 2
        self.interpolator = TimeConditionedMLP(
            input_dim=interp_input_dim,
            time_embed_dim=self.time_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.target_dim,
        )

        forecaster_input_dim = self.target_dim + (self.target_dim if self.condition_on_xt else 0)
        self.forecaster = TimeConditionedMLP(
            input_dim=forecaster_input_dim,
            time_embed_dim=self.time_embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.target_dim,
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

    def _interpolate(self, x_t, x_th, t_idx):
        # Eq. (2): I_phi(x_t, x_{t+h}, i) ~= x_{t+i}
        inp = torch.cat([x_t, x_th], dim=-1)
        t_norm = self._normalize_time(t_idx)
        pred = self.interpolator(inp, t_norm)
        if t_idx.ndim == 0:
            return x_t if t_idx.item() == 0 else pred
        mask = (t_idx == 0).unsqueeze(-1)
        return torch.where(mask, x_t, pred)

    def _forecast(self, s_n, t_idx, x_t=None):
        # Eq. (3): F_theta(s(n), i_n) ~= x_{t+h}
        if self.condition_on_xt and x_t is not None:
            inp = torch.cat([s_n, x_t], dim=-1)
        else:
            inp = s_n
        t_norm = self._normalize_time(t_idx)
        return self.forecaster(inp, t_norm)

    def _stage1_interpolator_loss(self, x_t, x_th, x_ti, i_idx):
        pred_ti = self._interpolate(x_t, x_th, i_idx)
        return F.mse_loss(pred_ti, x_ti, reduction="none").mean(dim=-1)

    def _stage2_forecaster_loss(self, x_t, x_th, schedule):
        bsz = x_t.shape[0]
        device = x_t.device
        n_idx = torch.randint(0, schedule.shape[0], (bsz,), device=device)
        i_n = schedule[n_idx]

        if self.stochastic_interpolator:
            self.interpolator.train()
        else:
            self.interpolator.eval()

        s_n = torch.where(i_n.unsqueeze(-1) == 0, x_t, self._interpolate(x_t, x_th, i_n))
        x_th_pred = self._forecast(s_n, i_n, x_t=x_t)
        loss = F.mse_loss(x_th_pred, x_th, reduction="none").mean(dim=-1)

        if self.lookahead_loss and schedule.shape[0] > 1:
            valid = (n_idx + 1 < schedule.shape[0])
            if valid.any():
                n_next = torch.clamp(n_idx + 1, max=schedule.shape[0] - 1)
                i_next = schedule[n_next]
                s_next = self._interpolate(x_t, x_th_pred, i_next)
                x_th_pred_next = self._forecast(s_next, i_next, x_t=x_t)
                loss_next = F.mse_loss(x_th_pred_next, x_th, reduction="none").mean(dim=-1)
                loss = torch.where(valid, 0.5 * (loss + loss_next), loss)

        return loss

    def loss(self, batch_data):
        x_t = batch_data.past_target_cdf[:, -1, :]
        x_th = batch_data.future_target_cdf[:, -1, :]

        schedule = self._build_schedule(self.num_diffusion_steps, x_t.device)

        stage = self.training_stage.lower()
        if stage == "interpolator":
            self._set_requires_grad(self.interpolator, True)
            self._set_requires_grad(self.forecaster, False)
            self.interpolator.train()
            self.forecaster.eval()

            if self.horizon < 2:
                return torch.zeros((), device=x_t.device)

            i_idx = torch.randint(1, self.horizon, (x_t.shape[0],), device=x_t.device).float()
            x_ti = batch_data.future_target_cdf[torch.arange(x_t.shape[0], device=x_t.device), i_idx.long() - 1]
            loss = self._stage1_interpolator_loss(x_t, x_th, x_ti, i_idx)
            return loss.mean()

        if stage == "forecaster":
            self._set_requires_grad(self.interpolator, False)
            self._set_requires_grad(self.forecaster, True)
            self.forecaster.train()
            loss = self._stage2_forecaster_loss(x_t, x_th, schedule)
            return loss.mean()

        if stage == "joint":
            self._set_requires_grad(self.interpolator, True)
            self._set_requires_grad(self.forecaster, True)
            self.interpolator.train()
            self.forecaster.train()

            if self.horizon < 2:
                return self._stage2_forecaster_loss(x_t, x_th, schedule).mean()

            i_idx = torch.randint(1, self.horizon, (x_t.shape[0],), device=x_t.device).float()
            x_ti = batch_data.future_target_cdf[torch.arange(x_t.shape[0], device=x_t.device), i_idx.long() - 1]
            loss_interp = self._stage1_interpolator_loss(x_t, x_th, x_ti, i_idx)
            loss_forecast = self._stage2_forecaster_loss(x_t, x_th, schedule)
            return (loss_interp + loss_forecast).mean()

        raise ValueError(f"Unknown training_stage: {self.training_stage}")

    def forecast(self, batch_data, num_samples=None):
        x_t = batch_data.past_target_cdf[:, -1, :]
        if num_samples is None:
            num_samples = 1
        if num_samples > 1:
            x_t = x_t.repeat_interleave(num_samples, dim=0)

        device = x_t.device
        schedule = self._build_schedule(self.sampling_steps, device)

        if self.stochastic_interpolator:
            self.interpolator.train()
        else:
            self.interpolator.eval()
        self.forecaster.eval()

        s_n = x_t
        x_th_pred = None
        for n in range(schedule.shape[0]):
            i_n = schedule[n].expand(x_t.shape[0])
            x_th_pred = self._forecast(s_n, i_n, x_t=x_t)
            if n + 1 >= schedule.shape[0]:
                break
            i_next = schedule[n + 1].expand(x_t.shape[0])
            interp_next = self._interpolate(x_t, x_th_pred, i_next)
            interp_curr = self._interpolate(x_t, x_th_pred, i_n)
            # Eq. (4): cold sampling update using interpolation increments
            s_n = interp_next - interp_curr + s_n

        if self.horizon == 1:
            forecasts = x_th_pred.unsqueeze(1)
        else:
            j_idx = torch.arange(1, self.horizon, device=device).float()
            j_rep = j_idx.unsqueeze(0).repeat(x_t.shape[0], 1).reshape(-1)
            x_t_rep = x_t.repeat_interleave(self.horizon - 1, dim=0)
            x_th_rep = x_th_pred.repeat_interleave(self.horizon - 1, dim=0)
            x_tj = self._interpolate(x_t_rep, x_th_rep, j_rep).view(
                x_t.shape[0], self.horizon - 1, self.target_dim
            )
            forecasts = torch.cat([x_tj, x_th_pred.unsqueeze(1)], dim=1)

        forecasts = forecasts.view(-1, num_samples, self.horizon, self.target_dim)
        return forecasts
