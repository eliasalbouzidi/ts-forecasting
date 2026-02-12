import os
import torch
import logging
from pathlib import Path
import sys
from probts.data import ProbTSDataModule
from probts.model.forecast_module import ProbTSForecastModule
from probts.callbacks import MemoryCallback, TimeCallback
from probts.utils import find_best_epoch
from probts.utils.evaluator import Evaluator
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from probts.utils.save_utils import save_exp_summary, save_csv

MULTI_HOR_MODEL = ['ElasTST', 'Autoformer']

import warnings
warnings.filterwarnings('ignore')

torch.set_float32_matmul_precision('high')

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ProbTSCli(LightningCLI):
    
    def add_arguments_to_parser(self, parser):
        data_to_model_link_args = [
            "scaler",
            "train_pred_len_list", 
        ]
        data_to_forecaster_link_args = [
            "target_dim",
            "history_length",
            "context_length",
            "prediction_length",
            "train_pred_len_list", 
            "lags_list",
            "freq",
            "time_feat_dim",
            "global_mean",
            "dataset"
        ]
        for arg in data_to_model_link_args:
            parser.link_arguments(f"data.data_manager.{arg}", f"model.{arg}", apply_on="instantiate")
        for arg in data_to_forecaster_link_args:
            parser.link_arguments(f"data.data_manager.{arg}", f"model.forecaster.init_args.{arg}", apply_on="instantiate")
        parser.add_argument(
            "--monitor_metric",
            type=str,
            default="auto",
            help=(
                "Metric to monitor for best checkpoint. "
                "Use 'auto' (default) or a metric name like 'CRPS', 'weighted_ND', 'MASE', "
                "'norm_CRPS', or full key like 'val_CRPS'."
            ),
        )
        parser.add_argument(
            "--wandb_run_name",
            type=str,
            default=None,
            help="Optional W&B run name. Defaults to the auto-generated tag.",
        )

    def init_exp(self):
        config_args = self.parser.parse_args()
        self.wandb_run_name = config_args.wandb_run_name
        self.run_config_paths = self._extract_run_config_paths(config_args)
        
        dl_suffix = "_dl" if getattr(self.datamodule.data_manager, "scaler_fit_on_full_data", False) else ""

        if self.datamodule.data_manager.multi_hor:
            assert self.model.forecaster.name in MULTI_HOR_MODEL, f"Only support multi-horizon setting for {MULTI_HOR_MODEL}"
            
            self.tag = "_".join([
                self.datamodule.data_manager.dataset,
                self.model.forecaster.name,
                'TrainCTX','-'.join([str(i) for i in self.datamodule.data_manager.train_ctx_len_list]),
                'TrainPRED','-'.join([str(i) for i in self.datamodule.data_manager.train_pred_len_list]),
                'ValCTX','-'.join([str(i) for i in self.datamodule.data_manager.val_ctx_len_list]),
                'ValPRED','-'.join([str(i) for i in self.datamodule.data_manager.val_pred_len_list]),
                'seed' + str(config_args.seed_everything)
            ]) + dl_suffix
        else:
            self.tag = "_".join([
                self.datamodule.data_manager.dataset,
                self.model.forecaster.name,
                'CTX' + str(self.datamodule.data_manager.context_length),
                'PRED' + str(self.datamodule.data_manager.prediction_length),
                'seed' + str(config_args.seed_everything)
            ]) + dl_suffix
        
        log.info(f"Root dir is {self.trainer.default_root_dir}, exp tag is {self.tag}")
        
        if not os.path.exists(self.trainer.default_root_dir):
            os.makedirs(self.trainer.default_root_dir)
            
        self.save_dict = f'{self.trainer.default_root_dir}/{self.tag}'
        if not os.path.exists(self.save_dict):
            os.makedirs(self.save_dict)

        if self.model.load_from_ckpt is not None:
            # if the checkpoint file is not assigned, find the best epoch in the current folder
            if '.ckpt' not in self.model.load_from_ckpt:
                _, best_ckpt = find_best_epoch(self.model.load_from_ckpt)
                print("find best ckpt ", best_ckpt)
                self.model.load_from_ckpt = os.path.join(self.model.load_from_ckpt, best_ckpt)
            
            log.info(f"Loading pre-trained checkpoint from {self.model.load_from_ckpt}")
            self.model = ProbTSForecastModule.load_from_checkpoint(
                self.model.load_from_ckpt,
                learning_rate=config_args.model.learning_rate,
                scaler=self.datamodule.data_manager.scaler,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                train_pred_len_list=self.datamodule.data_manager.train_pred_len_list,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                no_training=self.model.forecaster.no_training,
                sampling_weight_scheme=self.model.sampling_weight_scheme,
            )
        
        # Set callbacks
        self.memory_callback = MemoryCallback()
        self.time_callback = TimeCallback()
        
        callbacks = [
            self.memory_callback,
            self.time_callback
        ]
        
        if not self.model.forecaster.no_training:
            if self.datamodule.dataset_val is None:  # if the validation set is empty
                monitor = "train_loss"
            else:
                # not using reweighting scheme for loss
                if self.model.sampling_weight_scheme in ['none', 'fix']:
                    monitor = 'val_CRPS'
                else:
                    monitor = 'val_weighted_ND'
            monitor = self._resolve_monitor_metric(
                config_args.monitor_metric,
                monitor,
                has_val=self.datamodule.dataset_val is not None,
            )
            
            # Set callbacks
            monitor_token = monitor.replace("/", "_")
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=f'{self.save_dict}/ckpt',
                filename=f'{{epoch}}-{{{monitor_token}:.6f}}',
                every_n_epochs=1,
                monitor=monitor,
                mode='min',
                save_top_k=1,
                save_last=True,
                enable_version_counter=False
            )

            callbacks.append(self.checkpoint_callback)

        self.set_callbacks(callbacks)
        self._log_available_metrics()

    def _log_available_metrics(self):
        base_metrics = Evaluator().selected_metrics + ["loss"]
        base_metrics = sorted(set(base_metrics))
        log.info(
            "Available monitor metrics: %s (prefix with 'norm_' or 'val_' if needed)",
            ", ".join(base_metrics),
        )

    def _resolve_monitor_metric(self, requested, default_monitor, has_val):
        if requested == "auto":
            return default_monitor

        metric = requested.strip()
        if metric.startswith("test_"):
            log.warning("Monitor metric %s is test-only; using %s instead.", metric, default_monitor)
            return default_monitor

        if metric in ("loss", "train_loss", "val_loss"):
            if metric == "loss":
                return "val_loss" if has_val else "train_loss"
            if metric == "val_loss" and not has_val:
                log.warning("val_loss requested but no val set; using train_loss.")
                return "train_loss"
            return metric

        if metric.startswith(("val_", "train_")):
            if metric.startswith("val_") and not has_val:
                log.warning("%s requested but no val set; using train_%s.", metric, metric[4:])
                return f"train_{metric[4:]}"
            return metric

        stage = "val" if has_val else "train"
        return f"{stage}_{metric}"

    def set_callbacks(self, callbacks):
        # Replace built-in callbacks with custom callbacks
        custom_callbacks_name = [c.__class__.__name__ for c in callbacks]
        for c in self.trainer.callbacks:
            if c.__class__.__name__ in custom_callbacks_name:
                self.trainer.callbacks.remove(c)
        for c in callbacks:
            self.trainer.callbacks.append(c)
        for c in self.trainer.callbacks:
            if c.__class__.__name__ == "ModelSummary":
                self.model_summary_callback = c

    def set_fit_mode(self):
        tb_logger = TensorBoardLogger(
            save_dir=f'{self.save_dict}/logs',
            name=self.tag,
            version='fit'
        )
    
        self.wandb_logger = WandbLogger(
            project="probts",
            name=self.wandb_run_name or self.tag,
            save_dir=f'{self.save_dict}/logs',
        )
    
        self.trainer.loggers = [tb_logger, self.wandb_logger]
        self._upload_run_config_to_wandb()

    def _extract_run_config_paths(self, config_args):
        config_candidates = []

        cfg = getattr(config_args, "config", None)
        if cfg is not None:
            if isinstance(cfg, (list, tuple)):
                config_candidates.extend(cfg)
            else:
                config_candidates.append(cfg)

        # Fallback: parse explicit CLI flags if parser output does not expose config paths.
        argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            token = argv[i]
            if token in ("--config", "-c") and i + 1 < len(argv):
                config_candidates.append(argv[i + 1])
                i += 2
                continue
            if token.startswith("--config="):
                config_candidates.append(token.split("=", 1)[1])
            i += 1

        resolved = []
        seen = set()
        for c in config_candidates:
            p = Path(str(c)).expanduser()
            if not p.exists() or not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            resolved.append(rp)
        return resolved

    def _upload_run_config_to_wandb(self):
        if not hasattr(self, "wandb_logger"):
            return

        run = getattr(self.wandb_logger, "experiment", None)
        if run is None:
            return

        config_paths = getattr(self, "run_config_paths", [])
        if not config_paths:
            log.warning("No run config path detected; skipped W&B run-config upload.")
            return

        saved = 0
        for path in config_paths:
            try:
                run.save(str(path), policy="now")
                saved += 1
            except Exception as exc:
                log.warning("Failed to save run config file '%s' to W&B run files: %s", path, exc)

        if saved:
            log.info("Saved %d run config file(s) to W&B run files.", saved)
    
    def set_test_mode(self):
        csv_logger = CSVLogger(
            save_dir=f'{self.save_dict}/logs',
            name=self.tag,
            version='test'
        )
    
        if hasattr(self, "wandb_logger"):
            self.trainer.loggers = [csv_logger, self.wandb_logger]
        else:
            self.trainer.loggers = csv_logger


        if not self.model.forecaster.no_training:
            self.ckpt = self._resolve_test_checkpoint()
            if self.ckpt is None:
                log.warning(
                    "No checkpoint found for test-time reload. "
                    "Proceeding with in-memory model weights from the end of training."
                )
                return

            log.info(f"Loading checkpoint for test from {self.ckpt}")
            self.model = ProbTSForecastModule.load_from_checkpoint(
                self.ckpt,
                scaler=self.datamodule.data_manager.scaler,
                context_length=self.datamodule.data_manager.context_length,
                target_dim=self.datamodule.data_manager.target_dim,
                freq=self.datamodule.data_manager.freq,
                prediction_length=self.datamodule.data_manager.prediction_length,
                lags_list=self.datamodule.data_manager.lags_list,
                time_feat_dim=self.datamodule.data_manager.time_feat_dim,
                sampling_weight_scheme=self.model.sampling_weight_scheme,
            )

    def _resolve_test_checkpoint(self):
        # Some checkpoint policies (e.g. save_top_k=-1) may not populate best_model_path.
        candidates = [
            getattr(self.checkpoint_callback, "best_model_path", ""),
            getattr(self.checkpoint_callback, "last_model_path", ""),
        ]
        for ckpt_path in candidates:
            if ckpt_path and os.path.isfile(ckpt_path):
                return ckpt_path

        ckpt_dir = f"{self.save_dict}/ckpt"
        if os.path.isdir(ckpt_dir):
            _, best_ckpt = find_best_epoch(ckpt_dir)
            if best_ckpt is not None:
                ckpt_path = os.path.join(ckpt_dir, best_ckpt)
                if os.path.isfile(ckpt_path):
                    return ckpt_path
        return None

    def run(self):
        self.init_exp()
        
        if not self.model.forecaster.no_training:
            self.set_fit_mode()
            if self.datamodule.dataset_val is None:  # if the validation set is empty
                self.trainer.fit(model=self.model, train_dataloaders=self.datamodule.train_dataloader())
            else:
                self.trainer.fit(model=self.model, datamodule=self.datamodule)
            
            inference=False
        else:
            inference=True

        self.set_test_mode()
        self.trainer.test(model=self.model, datamodule=self.datamodule)
        
        save_exp_summary(self, inference=inference)
        
        ctx_len = self.datamodule.data_manager.context_length
        if self.datamodule.data_manager.multi_hor:
            ctx_len = ctx_len[0]

        save_csv(self.save_dict, self.model, ctx_len)


if __name__ == '__main__':
    cli = ProbTSCli(
        datamodule_class=ProbTSDataModule,
        model_class=ProbTSForecastModule,
        save_config_kwargs={"overwrite": True},
        run=False
    )
    cli.run()
