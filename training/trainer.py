from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp

try:
    from hydra.utils import instantiate
except ImportError:  # pragma: no cover
    def instantiate(*args, **kwargs):
        _ = args, kwargs
        raise ImportError("Missing `hydra-core`. Install with `bash scripts/setup_uv.sh train`.")

from .train_utils.distributed import get_machine_local_and_dist_rank
from .train_utils.freeze import freeze_modules
from .train_utils.general import (
    AverageMeter,
    DurationMeter,
    get_resume_checkpoint,
    model_summary,
    safe_makedirs,
    set_seeds,
)
from .train_utils.logging import setup_logging
from .train_utils.optimizer import construct_optimizers
from .train_utils.tb_writer import SummaryWriter


def _to_plain_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if hasattr(config, "items"):
        return {k: _to_plain_dict(v) for k, v in config.items()}
    return config


def _to_jax(data):
    if isinstance(data, dict):
        return {k: _to_jax(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        if not data:
            return data
        if isinstance(data[0], str):
            return data
        return type(data)(_to_jax(v) for v in data)
    if data is None or isinstance(data, str):
        return data
    return jnp.asarray(data)


def _replace_none_grads_with_zeros(grads, params):
    def _fix(g, p):
        if g is None:
            return jnp.zeros_like(p)
        return g

    return jax.tree_util.tree_map(_fix, grads, params, is_leaf=lambda x: x is None)


class Trainer:
    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] | None = None,
        cuda: Dict[str, bool] | None = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        _ = distributed, cuda, kwargs
        self._setup_env_variables(env_variables)
        self.start_time = time.time()

        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        self.accum_steps = int(accum_steps)
        self.max_epochs = int(max_epochs)
        self.mode = mode
        self.val_epoch_freq = int(val_epoch_freq)
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = int(seed_value)
        self.device = device

        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        self.rank = 0

        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(self.seed_value, self.max_epochs, self.distributed_rank)

        self._setup_components()
        self._setup_dataloaders()
        self.time_elapsed_meter = DurationMeter("Time Elapsed", None, ":.4f")

        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        self._setup_checkpoint_manager()
        self._load_checkpoint_if_any()

    def _setup_env_variables(self, env_variables_conf):
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = str(value)
        logging.info("Environment:\n%s", json.dumps(dict(os.environ), sort_keys=True, indent=2))

    def _setup_components(self):
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}

        try:
            self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        except Exception:
            self.tb_writer = SummaryWriter(str(Path(self.logging_conf.log_dir) / "tensorboard"))

        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False) if self.loss_conf is not None else None
        self.gradient_clipper = None
        if self.optim_conf is not None and getattr(self.optim_conf, "gradient_clip", None) is not None:
            self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)

        if self.optim_conf is not None and getattr(self.optim_conf, "frozen_module_names", None):
            self.model = freeze_modules(self.model, patterns=self.optim_conf.frozen_module_names)

        model_summary_path = Path(self.logging_conf.log_dir) / "model.txt"
        model_summary_path.write_text(model_summary(self.model), encoding="utf-8")

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None
        if self.mode in ["train", "val"] and self.data_conf.get("val", None) is not None:
            self.val_dataset = instantiate(self.data_conf.get("val"), _recursive_=False)
            self.val_dataset.seed = self.seed_value
        if self.mode in ["train"] and self.data_conf.get("train", None) is not None:
            self.train_dataset = instantiate(self.data_conf.get("train"), _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_checkpoint_manager(self):
        save_dir = Path(self.checkpoint_conf.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = save_dir
        save_freq = int(getattr(self.checkpoint_conf, "save_freq", 1) or 1)
        max_to_keep = getattr(self.checkpoint_conf, "max_to_keep", None)
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            keep_period=save_freq if save_freq > 1 else None,
            max_to_keep=max_to_keep,
            create=True,
            enable_async_checkpointing=True,
        )
        self.ckpt_manager = ocp.CheckpointManager(self.checkpoint_dir, options=options)

    def _build_checkpoint_payload(self, *, include_opt_state: bool = True):
        payload = {
            "epoch": int(self.epoch),
            "steps": {"train": int(self.steps.get("train", 0)), "val": int(self.steps.get("val", 0))},
            "params": nnx.state(self.model, nnx.Param),
            "time_elapsed": float(self.time_elapsed_meter.update()),
        }
        if include_opt_state and self.mode != "val" and hasattr(self, "optims"):
            payload["opt_state"] = self.optims.state
        return payload

    def _apply_restored_checkpoint(self, checkpoint: Dict[str, Any], *, source: str):
        if "params" in checkpoint:
            nnx.update(self.model, checkpoint["params"])
        if self.mode != "val" and "opt_state" in checkpoint and hasattr(self, "optims"):
            self.optims.state = checkpoint["opt_state"]
        self.epoch = int(checkpoint.get("epoch", self.epoch))
        steps = checkpoint.get("steps", self.steps)
        if isinstance(steps, dict):
            self.steps = {
                "train": int(steps.get("train", self.steps.get("train", 0))),
                "val": int(steps.get("val", self.steps.get("val", 0))),
            }
        logging.info("Resumed from checkpoint: %s", source)

    def _restore_from_manager(self, manager: ocp.CheckpointManager, *, source: str) -> bool:
        step = manager.latest_step()
        if step is None:
            return False
        template = self._build_checkpoint_payload(include_opt_state=(self.mode != "val"))
        checkpoint = manager.restore(step, args=ocp.args.StandardRestore(item=template, strict=False))
        self._apply_restored_checkpoint(checkpoint, source=f"{source} (step={step})")
        return True

    def _restore_from_manager_dir(self, ckpt_dir: Path, *, source: str) -> bool:
        ckpt_dir = ckpt_dir.expanduser().resolve()
        options = ocp.CheckpointManagerOptions(read_only=True, create=False)
        try:
            manager = ocp.CheckpointManager(ckpt_dir, options=options)
        except Exception:
            return False
        try:
            return self._restore_from_manager(manager, source=source)
        except Exception:
            return False
        finally:
            manager.close()

    def _load_checkpoint_if_any(self):
        resume_path = getattr(self.checkpoint_conf, "resume_checkpoint_path", None)
        if resume_path not in [None, "", "/YOUR/PATH/TO/CKPT"]:
            ckpt_path = Path(resume_path).expanduser().resolve()
            if ckpt_path.exists():
                if ckpt_path.is_dir():
                    if self._restore_from_manager_dir(ckpt_path, source=str(ckpt_path)):
                        return
                else:
                    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
                    checkpoint = checkpointer.restore(str(ckpt_path))
                    self._apply_restored_checkpoint(checkpoint, source=str(ckpt_path))
                    return

        if self._restore_from_manager(self.ckpt_manager, source=self.checkpoint_conf.save_dir):
            return

        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None or not Path(ckpt_path).exists():
            return
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        checkpoint = checkpointer.restore(ckpt_path)
        self._apply_restored_checkpoint(checkpoint, source=ckpt_path)

    def save_checkpoint(self, epoch: int):
        if self.mode == "val":
            return
        payload = self._build_checkpoint_payload(include_opt_state=True)
        saved = self.ckpt_manager.save(int(epoch), args=ocp.args.StandardSave(payload), force=True)
        self.ckpt_manager.wait_until_finished()
        if not saved:
            logging.warning("Checkpoint save skipped at epoch=%s", epoch)

    def _get_scalar_log_keys(self, phase: str):
        scalar_conf = getattr(self.logging_conf, "scalar_keys_to_log", None)
        if not scalar_conf:
            return []
        phase_conf = scalar_conf.get(phase, None)
        if phase_conf is None:
            return []
        keys = phase_conf.get("keys_to_log", phase_conf)
        return list(keys)

    def _write_loss_scalars(self, phase: str, loss_dict: Dict[str, jnp.ndarray], global_step: int):
        keys = self._get_scalar_log_keys(phase)
        if not keys:
            keys = list(loss_dict.keys())
        for key in keys:
            if key not in loss_dict:
                continue
            value = loss_dict[key]
            if value is None:
                continue
            try:
                scalar = float(value)
            except Exception:
                continue
            self.tb_writer.add_scalar(f"{phase}/{key}", scalar, global_step)

    def _compute_loss(self, batch):
        predictions = self.model(batch["images"])
        loss_dict = self.loss(predictions, batch)
        return loss_dict["objective"], loss_dict

    def _train_step(self, batch):
        batch = _to_jax(batch)

        def loss_fn(model):
            return self._compute_loss(batch)

        (objective, loss_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model)
        params = nnx.state(self.model, nnx.Param)
        grads = _replace_none_grads_with_zeros(grads, params)

        if self.gradient_clipper is not None:
            grads = self.gradient_clipper(grads)

        updates, self.optims.state = self.optims.optimizer.update(grads, self.optims.state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(self.model, new_params)
        return float(objective), loss_dict

    def _eval_step(self, batch):
        batch = _to_jax(batch)
        objective, loss_dict = self._compute_loss(batch)
        return float(objective), loss_dict

    def run(self):
        try:
            if self.mode == "train":
                self.run_train()
                self.run_val()
            elif self.mode == "val":
                self.run_val()
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        finally:
            self.close()

    def close(self):
        if hasattr(self, "ckpt_manager") and self.ckpt_manager is not None:
            try:
                self.ckpt_manager.wait_until_finished()
                self.ckpt_manager.close()
            except Exception:
                pass

    def run_train(self):
        if self.train_dataset is None:
            logging.info("No training dataset configured. Skipping.")
            return

        self.model.train()
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            train_loader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(train_loader)
            self.save_checkpoint(self.epoch)
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            self.epoch += 1

    def run_val(self):
        if self.val_dataset is None:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        self.model.eval()
        val_loader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(val_loader)
        self.model.train()

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("train_batch_time")
        objective_meter = AverageMeter("train_objective")
        end = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if self.limit_train_batches is not None and batch_idx >= int(self.limit_train_batches):
                break

            objective, loss_dict = self._train_step(batch)
            objective_meter.update(objective, n=1)
            batch_time.update(time.time() - end, n=1)
            end = time.time()

            step = self.steps["train"]
            self._write_loss_scalars("train", loss_dict, step)
            self.tb_writer.add_scalar("train/objective", objective, step)
            self.tb_writer.add_scalar("train/batch_time", batch_time.avg, step)
            self.steps["train"] += 1

        logging.info(
            "Epoch %s train done | objective=%.6f | step=%s",
            self.epoch,
            objective_meter.avg,
            self.steps["train"],
        )

    def val_epoch(self, val_loader):
        objective_meter = AverageMeter("val_objective")
        for batch_idx, batch in enumerate(val_loader):
            if self.limit_val_batches is not None and batch_idx >= int(self.limit_val_batches):
                break
            objective, loss_dict = self._eval_step(batch)
            objective_meter.update(objective, n=1)
            step = self.steps["val"]
            self._write_loss_scalars("val", loss_dict, step)
            self.tb_writer.add_scalar("val/objective", objective, step)
            self.steps["val"] += 1

        logging.info(
            "Epoch %s val done | objective=%.6f | step=%s",
            self.epoch,
            objective_meter.avg if objective_meter.count > 0 else math.nan,
            self.steps["val"],
        )
