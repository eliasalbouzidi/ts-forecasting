import torch
from torch import nn
from torch.nn import functional as F

from probts.model.forecaster import Forecaster
from probts.model.forecaster.prob_forecaster.dyffusion import Dyffusion


class AutoEncoderWithSkip(nn.Module):
    """
    AutoEncoder with skip connections for time series.
    The encoder extracts dynamics, which are passed to both Dyffusion and Decoder.
    This preserves system dynamics throughout the pipeline.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        # Encoder: input_dim -> latent_dim (extracts dynamics)
        encoder_layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(max(num_layers - 1, 0)):
            encoder_layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: (latent_dim + latent_dim) -> input_dim
        # Takes both: predicted latents + encoded dynamics
        decoder_input_dim = latent_dim * 2
        decoder_layers = [nn.Linear(decoder_input_dim, hidden_dim), nn.GELU()]
        for _ in range(max(num_layers - 1, 0)):
            decoder_layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to extract dynamics"""
        return self.encoder(x)
    
    def decode(self, z_pred, z_dynamics):
        """Decode predictions using both predicted latents and encoded dynamics"""
        combined = torch.cat([z_pred, z_dynamics], dim=-1)
        return self.decoder(combined)
    
    def forward(self, x):
        """Forward pass (not used in training, but kept for compatibility)"""
        z = self.encode(x)
        return self.decode(z, z)


class DyffusionAutoencoder(Forecaster):
    """
    Dyffusion with AutoEncoder that has skip connections for dynamics preservation.
    
    Pipeline:
    1. Encoder: past context → latent dynamics
    2. Dyffusion: learns to predict in latent space (informed by dynamics)
    3. Decoder: (predicted_latent + encoded_dynamics) → output
    
    The encoder extracts system dynamics, which flow through both Dyffusion and to the Decoder.
    """
    
    def __init__(
        self,
        latent_dim: int = 64,
        ae_hidden_dim: int = 128,
        ae_num_layers: int = 2,
        # Dyffusion parameters
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
        self.ae_hidden_dim = ae_hidden_dim
        self.ae_num_layers = ae_num_layers
        
        # Initialize AutoEncoder with skip connections
        self.autoencoder = AutoEncoderWithSkip(
            input_dim=self.target_dim,
            latent_dim=latent_dim,
            hidden_dim=ae_hidden_dim,
            num_layers=ae_num_layers,
        )
        
        # Initialize Dyffusion in latent space
        dyffusion_kwargs = kwargs.copy()
        dyffusion_kwargs.pop('target_dim')
        dyffusion_kwargs['target_dim'] = latent_dim  # Operate on latent space
        
        self.dyffusion = Dyffusion(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            num_diffusion_steps=num_diffusion_steps,
            sampling_steps=sampling_steps,
            training_stage=training_stage,
            lookahead_loss=lookahead_loss,
            condition_on_xt=condition_on_xt,
            stochastic_interpolator=stochastic_interpolator,
            **dyffusion_kwargs,
        )
    
    def loss(self, batch_data):
        """
        Compute loss using autoencoder + dyffusion with skip connections.
        
        1. Encode past context → extract dynamics
        2. Encode future target → extract target dynamics
        3. Dyffusion learns in latent space (informed by past dynamics)
        4. Decoder uses both prediction and past dynamics
        """
        # Encode past context to extract dynamics
        # batch_data.past_target_cdf: (B, L, target_dim)
        z_past_dynamics = self.autoencoder.encode(batch_data.past_target_cdf)  # (B, L, latent_dim)
        
        # Encode future target to extract target dynamics
        # batch_data.future_target_cdf: (B, H, target_dim)
        z_future_dynamics = self.autoencoder.encode(batch_data.future_target_cdf)  # (B, H, latent_dim)
        
        # Create a new batch_data with encoded values for dyffusion
        class EncodedBatchData:
            def __init__(self, past, future):
                self.past_target_cdf = past
                self.future_target_cdf = future
        
        z_batch_data = EncodedBatchData(z_past_dynamics, z_future_dynamics)
        
        # Compute loss in latent space using dyffusion
        return self.dyffusion.loss(z_batch_data)
    
    def forecast(self, batch_data, num_samples=None):
        """
        Generate forecasts using autoencoder + dyffusion with skip connections.
        
        1. Encode past context → extract dynamics (z_context_dynamics)
        2. Dyffusion in latent space predicts → z_pred
        3. Decoder uses (z_pred + z_context_dynamics) → final predictions
        """
        # Encode past context to extract dynamics
        z_context_dynamics = self.autoencoder.encode(batch_data.past_target_cdf)  # (B, L, latent_dim)
        
        # Use only the last timestep of past context for Dyffusion
        z_past_last = z_context_dynamics[:, -1:, :]  # (B, 1, latent_dim)
        
        # Create a new batch_data with encoded values for dyffusion
        class EncodedBatchData:
            def __init__(self, past):
                self.past_target_cdf = past
        
        z_batch_data = EncodedBatchData(z_past_last)
        
        # Generate forecasts in latent space
        z_forecasts = self.dyffusion.forecast(z_batch_data, num_samples=num_samples)  # (B, num_samples, H, latent_dim)
        
        # Average the context dynamics across time for skip connection
        z_context_avg = z_context_dynamics.mean(dim=1, keepdim=True)  # (B, 1, latent_dim)
        
        # Decode forecasts from latent space using skip connection
        # z_forecasts shape: (B, num_samples, H, latent_dim)
        B, num_samples_val, H, _ = z_forecasts.shape
        
        # Expand context dynamics to match forecast shape
        z_context_expanded = z_context_avg.expand(B, num_samples_val, H, self.latent_dim)  # (B, num_samples, H, latent_dim)
        
        # Concatenate predictions with dynamics for decoder
        z_combined = torch.cat([z_forecasts, z_context_expanded], dim=-1)  # (B, num_samples, H, 2*latent_dim)
        z_flat = z_combined.reshape(B * num_samples_val * H, 2 * self.latent_dim)
        
        # Decode using both prediction and dynamics
        forecasts_flat = self.autoencoder.decoder(z_flat)
        forecasts = forecasts_flat.reshape(B, num_samples_val, H, self.target_dim)
        
        return forecasts
