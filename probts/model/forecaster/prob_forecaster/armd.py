# probts/model/forecaster/prob_forecaster/armd.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import reduce 

from probts.model.forecaster import Forecaster
from probts.utils import repeat
from probts.model.nn.arch.armd_layers import LinearBackbone, default, extract, linear_beta_schedule, cosine_beta_schedule

class ARMD(Forecaster):
    def __init__(
            self,
            beta_schedule: str = 'linear',
            sampling_timesteps: int = None,
            loss_type: str = 'l1',
            w_grad: bool = True,
            # Configurable hyperparameters for devolution (b, c, d in Equation (5))
            b_param: float = 2.0,
            c_param: float = -1.0,
            d_param: float = 0.5,
            **kwargs
    ):
        super().__init__(**kwargs)
        
        self.autoregressive = False 
        
        self.seq_length = self.prediction_length
        self.feature_size = self.target_dim
        self.loss_type = loss_type

        # -------------------------------------------------------
        # Initialize Backbone (Devolution Network R)
        # -------------------------------------------------------
        self.model = LinearBackbone(
            n_feat=self.feature_size,
            n_channel=self.seq_length,
            timesteps=self.prediction_length,
            w_grad=w_grad,
            b_param=b_param,
            c_param=c_param,
            d_param=d_param
        ) 

        # -------------------------------------------------------
        # Diffusion Parameters
        # -------------------------------------------------------
        timesteps = self.prediction_length 
        self.num_timesteps = int(timesteps)
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # Standard DDPM helper variables
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # self.register_buffer ensures: Automatic Device Movement, State Dictionary saving, and No Gradients
        # Register the base schedules
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

        # Register pre-calculated terms for the forward/reverse equations
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())

        # Register loss weights
        self.register_buffer('loss_weight', (torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100).float())

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps

    # -------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------

    def predict_noise_from_start(self, x_t, t, x0):
        # Calculates the evolution trend z^t (analogous to noise) given x_t and x_0
        # Equation (3): z^t = (x^t - sqrt(alpha_bar) * x^0) / sqrt(1 - alpha_bar)
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, clip_x_start=False, training=False):
        # Predicts X^0 using the Devolution Network (Equation 5)
        x_start = self.model(x, t, training=training)
        
        if clip_x_start:
            x_start.clamp_(-1., 1.) 
        # Calculates predicted evolution trend z^t (Equation 6)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start
        
    # Forward Pass (corresponds to Equation (1) and Equation (2))
    def q_sample_sliding(self, future, history, t):
        """
        ARMD Sliding Mechanism: Forward Diffusion (Evolution) Process
        Formula: Eq (1) & (2) -> X^t_{1-t:T-t} = Slide(X^0_{1:T}, t)
        Derives intermediate states by sliding the series according to diffusion step t.
        """
        b = future.shape[0]
        
        # Ensure we only use the relevant immediate history
        relevant_history = history[:, -self.prediction_length:, :]
        
        # Concatenate: [B, 2 * Pred_Len, C]
        full_seq = torch.cat([relevant_history, future], dim=1)
        
        output_list = []
        total_len = full_seq.shape[1]
        
        for i in range(b):
            curr_t = t[i].item()
            # Slide logic: t=0 -> Future, t=T -> History
            start_idx = total_len - self.prediction_length - curr_t
            end_idx = total_len - curr_t
            output_list.append(full_seq[i, start_idx:end_idx, :])
            
        return torch.stack(output_list, dim=0)

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------

    def loss(self, batch_data):
        
        future_target = batch_data.future_target_cdf # X^0 (Target/Future)
        past_target = batch_data.past_target_cdf # X^T (History)
        
        # Scaling
        if self.use_scaling:
            self.get_scale(batch_data)
            
            # Access scaler attributes directly
            scale = self.scaler.scale
            
            # SAFEGUARD: Some scalers (TemporalScaler) might not have 'loc'.
            # If missing, assume 0 (centered).
            if hasattr(self.scaler, 'loc'):
                loc = self.scaler.loc
            elif hasattr(self.scaler, 'mean'):
                loc = self.scaler.mean
            else:
                loc = torch.zeros_like(scale)

            # Apply scaling
            future_target = (future_target - loc) / scale
            past_target = (past_target - loc) / scale

        # Time Sampling
        b = future_target.shape[0]
        device = future_target.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # Generate Intermediate State X^t via Sliding
        x_t = self.q_sample_sliding(future_target, past_target, t)
        
        # Model Prediction
        model_out = self.model(x_t, t, training=True)
        
        # Calculate Loss
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

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        return loss.mean()

    # -------------------------------------------------------
    # Inference / Forecasting
    # -------------------------------------------------------

    def forecast(self, batch_data, num_samples=None):
        past_target = batch_data.past_target_cdf
        
        # Initialize default scaling params
        loc = 0.0
        scale = 1.0

        if self.use_scaling:
            self.get_scale(batch_data)
            scale = self.scaler.scale
            
            # SAFEGUARD: Match logic in loss()
            if hasattr(self.scaler, 'loc'):
                loc = self.scaler.loc
            elif hasattr(self.scaler, 'mean'):
                loc = self.scaler.mean
            else:
                loc = torch.zeros_like(scale)
            
            past_target = (past_target - loc) / scale

        # Start with History (Final State X^T)
        curr_img = past_target[:, -self.prediction_length:, :]
        
        if num_samples is not None and num_samples > 1:
            curr_img = repeat(curr_img, num_samples, dim=0)

        # Sampling Loop (Reverse Devolution Process)
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 
        
        device = curr_img.device
        
        for time, time_next in time_pairs: 
            b = curr_img.shape[0]
            t_cond = torch.full((b,), time, device=device, dtype=torch.long)
            
            # Predict X^0 using the devolution network
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

        forecasts = curr_img
        
        # Descaling
        if self.use_scaling:
             if num_samples is not None and num_samples > 1:
                 # Expand loc/scale to match samples
                 if isinstance(loc, torch.Tensor):
                     loc_rep = repeat(loc, num_samples, dim=0)
                 else:
                     loc_rep = loc
                     
                 if isinstance(scale, torch.Tensor):
                     scale_rep = repeat(scale, num_samples, dim=0)
                 else:
                     scale_rep = scale
                     
                 forecasts = forecasts * scale_rep + loc_rep
             else:
                 forecasts = forecasts * scale + loc

        if num_samples is None: 
            num_samples = 1
            
        forecasts = forecasts.reshape(-1, num_samples, self.prediction_length, self.target_dim)
        
        return forecasts