import numpy as np
import torch
from torch import optim
from typing import Dict, List, Optional
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import sys

from probts.data import ProbTSBatchData
from probts.data.data_utils.data_scaler import Scaler
from probts.model.forecaster import Forecaster
from probts.utils.evaluator import Evaluator
from probts.utils.metrics import *
from probts.utils.save_utils import update_metrics, calculate_weighted_average, load_checkpoint, get_hor_str
from probts.utils.utils import init_class_helper

DEFAULT_WANDB_REPORT_METRICS = ["loss", "MSE", "MAE", "sMAPE", "CRPS", "DTW"]
DEFAULT_WANDB_METRIC_VIEWS = ["norm"]

def get_weights(sampling_weight_scheme, max_hor):
    '''
    return: w [max_hor]
    '''
    if sampling_weight_scheme == 'random':
        i_array = np.linspace(1 + 1e-5, max_hor - 1e-3, max_hor)
        w = (1 / max_hor) * (np.log(max_hor) - np.log(i_array))
    elif sampling_weight_scheme == 'const':
        w = np.array([1 / max_hor] * max_hor)
    elif sampling_weight_scheme == 'none':
        return None
    else:
        raise ValueError(f"Invalid sampling scheme {sampling_weight_scheme}.")
    
    return torch.tensor(w)


class ProbTSForecastModule(pl.LightningModule):
    def __init__(
        self,
        forecaster: Forecaster,
        scaler: Scaler = None,
        train_pred_len_list: list = None,
        num_samples: int = 100,
        learning_rate: float = 1e-3,
        quantiles_num: int = 10,
        wandb_report_metrics: Optional[List[str]] = None,
        wandb_metric_views: Optional[List[str]] = None,
        wandb_include_sum: bool = False,
        wandb_log_forecast_plots: bool = True,
        wandb_forecast_plot_max_dims: int = 0,
        wandb_forecast_plot_view: str = "both",
        load_from_ckpt: str = None,
        sampling_weight_scheme: str = 'none',
        optimizer_config = None,
        lr_scheduler_config = None,
        **kwargs
    ):
        super().__init__()
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.load_from_ckpt = load_from_ckpt
        self.train_pred_len_list = train_pred_len_list
        self.forecaster = forecaster
        self.optimizer_config = optimizer_config
        self.scheduler_config = lr_scheduler_config
        
        if self.optimizer_config is not None:
            print("optimizer config: ", self.optimizer_config)
            
        if self.scheduler_config is not None:
            print("lr_scheduler config: ", self.scheduler_config)
        
        self.scaler = scaler
        self.evaluator = Evaluator(quantiles_num=quantiles_num)
        if wandb_report_metrics is None:
            self.wandb_report_metrics = list(DEFAULT_WANDB_REPORT_METRICS)
        else:
            self.wandb_report_metrics = list(wandb_report_metrics)
        if wandb_metric_views is None:
            self.wandb_metric_views = list(DEFAULT_WANDB_METRIC_VIEWS)
        else:
            self.wandb_metric_views = [str(v).strip().lower() for v in wandb_metric_views]
        self.wandb_include_sum = bool(wandb_include_sum)
        self.wandb_log_forecast_plots = bool(wandb_log_forecast_plots)
        self.wandb_forecast_plot_max_dims = int(wandb_forecast_plot_max_dims)
        self.wandb_forecast_plot_view = str(wandb_forecast_plot_view).strip().lower()
        if self.wandb_forecast_plot_view not in {"denorm", "norm", "both"}:
            self.wandb_forecast_plot_view = "both"
        self._test_plot_payload = None
        
        # init the parapemetr for sampling
        self.sampling_weight_scheme = sampling_weight_scheme
        print(f'sampling_weight_scheme: {sampling_weight_scheme}')
        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(self, checkpoint_path, scaler=None, learning_rate=None, no_training=False, **kwargs):
        model = load_checkpoint(self, checkpoint_path, scaler=scaler, learning_rate=learning_rate, no_training=no_training, **kwargs)
        return model

    def training_forward(self, batch_data):
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        batch_data.future_target_cdf = self.scaler.transform(batch_data.future_target_cdf)
        loss = self.forecaster.loss(batch_data)

        if len(loss.shape) > 1:
            loss_weights = get_weights(self.sampling_weight_scheme, loss.shape[1])
            loss = (loss_weights.detach().to(loss.device).unsqueeze(0).unsqueeze(-1) * loss).sum(dim=1)
            loss = loss.mean()
        
        return loss

    def training_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=False)
        self._log_metrics_all({"train_loss": float(loss.detach().cpu())})
        return loss

    def evaluate(self, batch, stage='',dataloader_idx=None):
        batch_data = ProbTSBatchData(batch, self.device)
        pred_len = batch_data.future_target_cdf.shape[1]
        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]

        norm_past_data = self.scaler.transform(batch_data.past_target_cdf)
        norm_future_data = self.scaler.transform(batch_data.future_target_cdf)
        self.batch_size.append(orin_past_data.shape[0])
        
        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)[:,:, :pred_len]
        
        # Calculate denorm metrics
        denorm_forecasts = self.scaler.inverse_transform(forecasts)
        metrics = self.evaluator(orin_future_data, denorm_forecasts, past_data=orin_past_data, freq=self.forecaster.freq)
        self.metrics_dict = update_metrics(metrics, stage, target_dict=self.metrics_dict)
        
        # Calculate norm metrics
        norm_metrics = self.evaluator(norm_future_data, forecasts, past_data=norm_past_data, freq=self.forecaster.freq)
        self.metrics_dict = update_metrics(norm_metrics, stage, 'norm', target_dict=self.metrics_dict)
        
        l = orin_future_data.shape[1]
        
        if stage != 'test' and self.sampling_weight_scheme not in ['fix', 'none']:
            loss_weights = get_weights('random', l)
        else:
            loss_weights = None

        hor_metrics = self.evaluator(
            orin_future_data,
            denorm_forecasts,
            past_data=orin_past_data,
            freq=self.forecaster.freq,
            loss_weights=loss_weights,
        )
        hor_norm_metrics = self.evaluator(
            norm_future_data,
            forecasts,
            past_data=norm_past_data,
            freq=self.forecaster.freq,
            loss_weights=loss_weights,
        )
        
        if stage == 'test':
            hor_str = get_hor_str(self.forecaster.prediction_length, dataloader_idx)
            if hor_str not in self.hor_metrics:
                self.hor_metrics[hor_str] = {}

            
            self.hor_metrics[hor_str] = update_metrics(hor_metrics, stage, target_dict=self.hor_metrics[hor_str])
            self.hor_metrics[hor_str] = update_metrics(
                hor_norm_metrics,
                stage,
                key='norm',
                target_dict=self.hor_metrics[hor_str],
            )
            if self.wandb_log_forecast_plots and self._test_plot_payload is None:
                plot_idx = orin_past_data.shape[0] // 2
                self._test_plot_payload = {
                    "denorm_past": orin_past_data[plot_idx].detach().cpu().numpy(),
                    "denorm_future": orin_future_data[plot_idx].detach().cpu().numpy(),
                    "denorm_forecast_samples": denorm_forecasts[plot_idx].detach().cpu().numpy(),
                    "norm_past": norm_past_data[plot_idx].detach().cpu().numpy(),
                    "norm_future": norm_future_data[plot_idx].detach().cpu().numpy(),
                    "norm_forecast_samples": forecasts[plot_idx].detach().cpu().numpy(),
                }

        return hor_metrics

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            batch_data = ProbTSBatchData(batch, self.device)
            val_loss = self.training_forward(batch_data)
        self.log("val_loss", val_loss, prog_bar=True, logger=False)
        self._log_metrics_all({"val_loss": float(val_loss.detach().cpu())})
        metrics = self.evaluate(batch, stage='val',dataloader_idx=dataloader_idx)
        return metrics


    def on_validation_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.batch_size = []

    def on_validation_epoch_end(self):
        avg_metrics = calculate_weighted_average(self.metrics_dict, self.batch_size)
        self.log_dict(avg_metrics, prog_bar=True, logger=False)
        self._log_metrics_all(avg_metrics)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        metrics = self.evaluate(batch, stage='test',dataloader_idx=dataloader_idx)
        return metrics

    def on_test_epoch_start(self):
        self.metrics_dict = {}
        self.hor_metrics = {}
        self.avg_metrics = {}
        self.avg_hor_metrics = {}
        self.batch_size = []
        self._test_plot_payload = None

    def on_test_epoch_end(self):
        if len(self.hor_metrics) > 0:
            for hor_str, metric in self.hor_metrics.items():
                self.avg_hor_metrics[hor_str] = calculate_weighted_average(metric, batch_size=self.batch_size)
                self.avg_metrics.update(calculate_weighted_average(metric, batch_size=self.batch_size, hor=hor_str+'_'))
        else:
            self.avg_metrics = calculate_weighted_average(self.metrics_dict, self.batch_size)
        
        if isinstance(self.forecaster.prediction_length, int) or len(self.forecaster.prediction_length) < 2:
            self.log_dict(self.avg_metrics, logger=False)
            self._log_metrics_all(self.avg_metrics)
        self._log_wandb_forecast_plots()

    def predict_step(self, batch, batch_idx):
        batch_data = ProbTSBatchData(batch, self.device)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)
        return forecasts

    def configure_optimizers(self):
        if self.optimizer_config is None:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = init_class_helper(self.optimizer_config['class_name'])
            params = self.optimizer_config['init_args']
            optimizer = optimizer(self.parameters(), **params)
        
        if self.scheduler_config is not None:
            scheduler = init_class_helper(self.scheduler_config['class_name'])
            params = self.scheduler_config['init_args']
            scheduler = scheduler(optimizer=optimizer, **params)
            
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": None,
            }

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return optimizer

    def _log_metrics_all(self, metrics: Dict[str, float]):
        if not hasattr(self, "trainer") or self.trainer is None:
            return
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb_metrics = self._format_wandb_metrics(metrics)
                wandb_metrics = self._filter_wandb_metrics(wandb_metrics)
                logger.log_metrics(wandb_metrics, step=self._get_wandb_safe_step(logger))
            else:
                metrics_with_meta = dict(metrics)
                metrics_with_meta["epoch"] = int(self.current_epoch)
                metrics_with_meta["step"] = int(self.global_step)
                logger.log_metrics(metrics_with_meta, step=self.global_step)

    def _get_wandb_safe_step(self, logger: WandbLogger) -> int:
        step = int(self.global_step)
        try:
            run_step = int(getattr(logger.experiment, "step", step))
            return max(step, run_step)
        except Exception:
            return step

    def _filter_wandb_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        configured = {m.strip() for m in self.wandb_report_metrics if isinstance(m, str) and m.strip()}
        views = {v for v in self.wandb_metric_views if v in {"denorm", "norm"}}

        if not configured:
            return metrics
        if not views:
            views = set(DEFAULT_WANDB_METRIC_VIEWS)

        filtered = {}
        for key, value in metrics.items():
            leaf_name = key.split("/")[-1]
            is_sum = leaf_name.endswith("-Sum")
            if is_sum and not self.wandb_include_sum:
                continue
            base_name = leaf_name[:-4] if is_sum else leaf_name
            if base_name == "loss":
                if base_name in configured:
                    filtered[key] = value
                continue

            is_norm = "/norm/" in f"/{key}/"
            if is_norm and "norm" not in views:
                continue
            if (not is_norm) and "denorm" not in views:
                continue

            if base_name in configured:
                filtered[key] = value
        return filtered

    def _format_wandb_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        formatted = {}
        for key, value in metrics.items():
            parts = key.split("_")
            if len(parts) >= 2 and parts[0].isdigit():
                hor = parts[0]
                stage = parts[1]
                rest = parts[2:]
                if rest[:1] == ["norm"]:
                    rest = rest[1:]
                    new_key = f"{stage}/{hor}/norm/" + "_".join(rest)
                else:
                    new_key = f"{stage}/{hor}/" + "_".join(rest)
            elif parts[0] in ("train", "val", "test"):
                stage = parts[0]
                rest = parts[1:]
                if rest[:1] == ["norm"]:
                    rest = rest[1:]
                    new_key = f"{stage}/norm/" + "_".join(rest)
                else:
                    new_key = f"{stage}/" + "_".join(rest)
            else:
                new_key = key
            formatted[new_key] = value
        return formatted

    def _log_wandb_forecast_plots(self):
        if not self.wandb_log_forecast_plots or self._test_plot_payload is None:
            return
        if not hasattr(self, "trainer") or self.trainer is None:
            return

        try:
            import wandb
        except Exception:
            return

        plot_views = ["denorm", "norm"] if self.wandb_forecast_plot_view == "both" else [self.wandb_forecast_plot_view]

        for plot_view in plot_views:
            if plot_view == "norm":
                past = self._test_plot_payload["norm_past"]  # [history_length, target_dim]
                future = self._test_plot_payload["norm_future"]  # [prediction_length, target_dim]
                forecast_samples = self._test_plot_payload["norm_forecast_samples"]  # [num_samples, prediction_length, target_dim]
            else:
                past = self._test_plot_payload["denorm_past"]  # [history_length, target_dim]
                future = self._test_plot_payload["denorm_future"]  # [prediction_length, target_dim]
                forecast_samples = self._test_plot_payload["denorm_forecast_samples"]  # [num_samples, prediction_length, target_dim]

            pred_median = np.quantile(forecast_samples, 0.5, axis=0)
            pred_p10 = np.quantile(forecast_samples, 0.1, axis=0)
            pred_p90 = np.quantile(forecast_samples, 0.9, axis=0)

            target_dim = past.shape[-1]
            if self.wandb_forecast_plot_max_dims > 0:
                max_dims = min(target_dim, self.wandb_forecast_plot_max_dims)
            else:
                max_dims = target_dim

            x_hist = np.arange(past.shape[0])
            x_future = np.arange(past.shape[0], past.shape[0] + future.shape[0])
            images = []

            for dim in range(max_dims):
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(x_hist, past[:, dim], label="history", color="black", linewidth=1.5)
                ax.plot(x_future, future[:, dim], label="future_truth", color="tab:blue", linewidth=1.5)
                ax.plot(x_future, pred_median[:, dim], label="forecast_median", color="tab:orange", linewidth=1.8)
                ax.fill_between(
                    x_future,
                    pred_p10[:, dim],
                    pred_p90[:, dim],
                    color="tab:orange",
                    alpha=0.2,
                    label="forecast_p10_p90",
                )
                ax.axvline(x=past.shape[0] - 1, color="gray", linestyle="--", linewidth=1)
                ax.set_title(f"Test Forecast ({plot_view}) - dim {dim}")
                ax.set_xlabel("time")
                ax.set_ylabel("value")
                ax.legend(loc="best")
                ax.grid(alpha=0.2)
                fig.tight_layout()
                images.append(wandb.Image(fig, caption=f"dim={dim}"))
                plt.close(fig)

            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.log(
                        {f"test/forecast_plots_{plot_view}": images},
                        step=self._get_wandb_safe_step(logger),
                    )
