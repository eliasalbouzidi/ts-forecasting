# probts/model/forecaster/prob_forecaster/armd.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from functools import partial
from einops import reduce 

from probts.model.forecaster import Forecaster
from probts.utils import repeat

from probts.model.nn.arch.armd_layers import LinearBackbone, default, extract, linear_beta_schedule, cosine_beta_schedule

class ARMD(Forecaster):
    def __init__(
            self,
            beta_schedule: str = 'cosine',
            sampling_timesteps: int = None,
            loss_type: str = 'l1',
            w_grad: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        # ARMD is "non-autoregressive" in generation (generates whole sequence at once).
        self.autoregressive = False 
        
        self.seq_length = self.prediction_length
        self.feature_size = self.target_dim
        self.loss_type = loss_type

        # -------------------------------------------------------
        # Initialize Backbone
        # -------------------------------------------------------
        self.model = LinearBackbone(
            n_feat=self.feature_size,
            n_channel=self.seq_length,
            timesteps=self.prediction_length,
            w_grad=w_grad
        )

        # -------------------------------------------------------
        # Diffusion Parameters
        # -------------------------------------------------------
        timesteps = self.prediction_length # ARMD sets steps = prediction length
        self.num_timesteps = int(timesteps)
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Register buffers
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)).float())
        self.register_buffer('posterior_mean_coef1', (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).float())
        self.register_buffer('posterior_mean_coef2', ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)).float())

        # Loss weighting
        self.register_buffer('loss_weight', (torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100).float())

        # Sampling settings
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps
        self.eta = 0. # DDIM eta

    # -------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, clip_x_start=False, training=False):
        # In ARMD, the model outputs the predicted x_start (x0) directly
        x_start = self.model(x, t, training=training)
        
        if clip_x_start:
            x_start.clamp_(-1., 1.) 
            
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def q_sample_sliding(self, future, history, t):
        """
        ARMD Sliding Mechanism:
        Constructs the intermediate state X^t by sliding the window.
        x_start (future) is at t=0.
        target (history) is at t=T.
        """
        # Batch size
        b = future.shape[0]
        
        # Concatenate History and Future: [B, Hist_Len + Pred_Len, C]
        full_seq = torch.cat([history, future], dim=1)
        
        output_list = []
        for i in range(b):
            curr_t = t[i].item()
            total_len = full_seq.shape[1]
            # Slide logic: 
            # t=0 -> Future (End of seq)
            # t=T -> History (Start of seq, shifted by Prediction Length)
            start_idx = total_len - self.prediction_length - curr_t
            end_idx = total_len - curr_t
            
            # Safety check
            if start_idx < 0:
                start_idx = 0
                end_idx = self.prediction_length
                
            output_list.append(full_seq[i, start_idx:end_idx, :])
            
        return torch.stack(output_list, dim=0)

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    def loss(self, batch_data):
        if self.use_scaling:
            self.get_scale(batch_data)
        
        # Get Data
        future_target = batch_data.future_target_cdf 
        past_target = batch_data.past_target_cdf 
        
        if self.use_scaling:
            future_target = (future_target - self.scaler.loc) / self.scaler.scale
            past_target = (past_target - self.scaler.loc) / self.scaler.scale

        # Sample time steps
        b = future_target.shape[0]
        device = future_target.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Generate Intermediate State X^t via Sliding
        x_t = self.q_sample_sliding(future_target, past_target, t)
        
        # The Network predicts X^0 (Future) from X^t
        model_out = self.model(x_t, t, training=True)
        
        # Calculate Loss targets
        alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        target_noise = (x_t - future_target * alpha) / (minus_alpha + 1e-8)
        pred_noise = (x_t - model_out * alpha) / (minus_alpha + 1e-8)

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_noise, target_noise, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_noise, target_noise, reduction='none')
        else:
            loss = F.l1_loss(pred_noise, target_noise, reduction='none')

        # Apply weighting
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        return loss.mean()

    # -------------------------------------------------------
    # Inference / Forecasting
    # -------------------------------------------------------

    def forecast(self, batch_data, num_samples=None):
        if self.use_scaling:
            self.get_scale(batch_data)
        
        # Start with History (Final State X^T)
        past_target = batch_data.past_target_cdf
        
        if self.use_scaling:
            past_target = (past_target - self.scaler.loc) / self.scaler.scale

        # The starting point for sampling is the last 'pred_len' of history
        curr_img = past_target[:, -self.prediction_length:, :]
        
        # Handle num_samples
        if num_samples is not None and num_samples > 1:
            curr_img = repeat(curr_img, num_samples, dim=0)

        # Sampling Loop (Reverse Process)
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 
        
        device = curr_img.device
        
        # Loop
        for time, time_next in time_pairs: 
            b = curr_img.shape[0]
            t_cond = torch.full((b,), time, device=device, dtype=torch.long)
            
            # Predict x_0 (future)
            pred_noise, x_start = self.model_predictions(curr_img, t_cond, clip_x_start=False, training=False)
            
            if time_next < 0:
                curr_img = x_start
                continue
            
            # Deterministic Sampling (Sigma=0)
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = 0 
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            curr_img = x_start * alpha_next.sqrt() + c * pred_noise 

        # Final Prediction
        forecasts = curr_img
        
        if self.use_scaling:
             # Descale
             if num_samples is not None and num_samples > 1:
                 loc = repeat(self.scaler.loc, num_samples, dim=0)
                 scale = repeat(self.scaler.scale, num_samples, dim=0)
                 forecasts = forecasts * scale + loc
             else:
                 forecasts = forecasts * self.scaler.scale + self.scaler.loc

        # Reshape to [Batch, Samples, Pred_Len, Dim]
        if num_samples is None: 
            num_samples = 1
            
        forecasts = forecasts.reshape(-1, num_samples, self.prediction_length, self.target_dim)
        
        return forecasts