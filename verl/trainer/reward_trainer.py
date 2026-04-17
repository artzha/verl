# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward model trainer using pairwise Bradley-Terry loss on the packed RM dataset.

Designed to make minimal changes to SFTTrainer.  Key differences:
  1. Uses RMDataset (via config custom_cls) instead of MultiTurnSFTDataset.
  2. Uses a custom RMTensorCollator that interleaves chosen/rejected sequences:
       [chosen_0, rejected_0, chosen_1, rejected_1, ...]
     This guarantees that any micro-batch of even size contains complete pairs.
  3. Injects a synthetic loss_mask = attention_mask so the engine can compute
     batch_num_tokens (required by forward_backward_batch).
  4. Replaces sft_loss with rm_loss (Bradley-Terry on implicit log-prob rewards).

Usage:
    torchrun --nproc_per_node=<N> -m verl.trainer.reward_trainer \\
        --config-path <path> \\
        --config-name motion_reward_trainer \\
        data.train_files=data/unified_motion_rm_v1/parquet/train/train.parquet \\
        data.val_files=data/unified_motion_rm_v1/parquet/val/val.parquet \\
        model.path=Qwen/Qwen3-VL-2B-Instruct
"""

import logging
import os
import warnings
from functools import partial

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import torch
import torch.nn.functional as F
import torch.distributed
from omegaconf import OmegaConf
from tensordict.tensorclass import NonTensorData
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode, RMTensorCollator
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

from verl.trainer.sft_trainer import SFTTrainer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_RM_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def rm_loss(config, model_output, data, dp_group=None):
    """Bradley-Terry ranking loss on scalar rewards from the reward head.

    The batch must be in interleaved format:
        [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    so that ``rewards[::2]`` are chosen and ``rewards[1::2]`` are rejected.

    ``model_output["rewards"]`` is a 1-D tensor of shape ``(2*N,)`` produced
    by ``FSDPEngineWithRewardHead`` — one scalar per sequence, extracted from
    the last valid token's hidden state via the learned ``score`` head.

    Args:
        config:         Unused (kept for signature parity with sft_loss).
        model_output:   Dict with key ``rewards`` – tensor of shape ``(2*N,)``.
        data:           TensorDict for the micro-batch.
        dp_group:       Unused.

    Returns:
        (loss, metrics): scalar Bradley-Terry loss and a dict of RM diagnostics.
    """
    rewards = model_output["rewards"]  # (2*N,) – scalar reward per sequence
    assert rewards.dim() == 1, f"Expected 1-D rewards tensor, got shape {rewards.shape}"
    assert rewards.shape[0] % 2 == 0, (
        f"rewards must have even length (interleaved chosen/rejected), got {rewards.shape[0]}"
    )

    # Interleaved pairing: even = chosen, odd = rejected
    chosen_rewards = rewards[::2]    # (N,)
    rejected_rewards = rewards[1::2] # (N,)

    pair_loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    pair_loss_mask = data.get("pair_loss_mask", None)
    if pair_loss_mask is None:
        pair_mask = torch.ones_like(pair_loss, dtype=pair_loss.dtype, device=pair_loss.device)
    else:
        pair_loss_mask = pair_loss_mask.to(device=rewards.device, dtype=pair_loss.dtype)
        assert pair_loss_mask.shape[0] == rewards.shape[0], (
            "pair_loss_mask must have shape (2*N,) matching rewards. "
            f"Got {tuple(pair_loss_mask.shape)} vs rewards {tuple(rewards.shape)}."
        )
        pair_mask = torch.minimum(pair_loss_mask[::2], pair_loss_mask[1::2])

    denom = torch.clamp(pair_mask.sum(), min=1.0)
    loss = (pair_loss * pair_mask).sum() / denom

    with torch.no_grad():
        margin = ((chosen_rewards - rejected_rewards) * pair_mask).sum() / denom
        accuracy = ((chosen_rewards > rejected_rewards).float() * pair_mask).sum() / denom

    metrics = {
        "rm/chosen_reward": chosen_rewards.detach().mean().item(),
        "rm/rejected_reward": rejected_rewards.detach().mean().item(),
        "rm/reward_margin": margin.item(),
        "rm/accuracy": accuracy.item(),
        "rm/effective_pairs": pair_mask.detach().sum().item(),
    }
    return loss, metrics

# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def create_rm_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create an RMDataset, honouring ``custom_cls`` overrides in data_config."""
    from omegaconf import ListConfig

    if not isinstance(data_paths, (list, ListConfig)):
        data_paths = [data_paths]

    custom_cls_cfg = data_config.get("custom_cls", {})
    if custom_cls_cfg.get("path", None):
        from verl.utils.import_utils import load_extern_object
        dataset_cls = load_extern_object(custom_cls_cfg.path, custom_cls_cfg.name)
    else:
        from verl.utils.dataset.rm_dataset import RMDataset
        dataset_cls = RMDataset

    return dataset_cls(
        parquet_files=data_paths,
        tokenizer=tokenizer,
        config=data_config,
        processor=processor,
        max_samples=max_samples,
    )

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RewardTrainer(SFTTrainer):
    """Reward model trainer.

    Inherits the full training loop from SFTTrainer and overrides only:
      - ``_build_dataset``    – uses RMDataset
      - ``_build_engine``     – uses rm_loss instead of sft_loss
      - ``_build_dataloader`` – uses RMTensorCollator and no_padding mode
      - ``fit``               – adjusts meta_info and metric keys for RM
    """

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor

        self.train_dataset = create_rm_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        self.max_pairs_per_sample = int(getattr(self.train_dataset, "max_pairs_per_sample", 1))
        if config.data.val_files:
            self.val_dataset = create_rm_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
            val_max_pairs = int(getattr(self.val_dataset, "max_pairs_per_sample", self.max_pairs_per_sample))
            if val_max_pairs != self.max_pairs_per_sample:
                raise ValueError(
                    f"Train/val max_pairs_per_sample mismatch: train={self.max_pairs_per_sample}, "
                    f"val={val_max_pairs}. Re-pack datasets with a consistent max_pairs."
                )
        else:
            self.val_dataset = None

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        # Import to trigger @EngineRegistry.register for reward_model
        import verl.workers.engine.fsdp.transformer_impl  # noqa: F401

        self.loss_fn = partial(rm_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="reward_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )
        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.engine = self.training_client.engine

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _build_dataloader(self):
        config = self.config
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        # global_batch_size is the number of *samples* per gradient step.
        # RMTensorCollator flattens each sample to 2 * max_pairs_per_sample sequences.
        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size

        self.collate_fn = RMTensorCollator()
        dataloader_num_workers = int(config.data.get("dataloader_num_workers", 12))
        pin_memory = bool(config.data.get("pin_memory", False))
        pin_memory_device = device_name if pin_memory else ""

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=dataloader_num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            pin_memory_device=pin_memory_device,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=dataloader_num_workers,
                pin_memory=pin_memory,
                drop_last=True,
                pin_memory_device=pin_memory_device,
            )
        else:
            self.val_dataloader = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        if is_logging:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step
        last_valid_metric = None
        early_stop_patience = int(getattr(self.config.trainer, "early_stop_patience", 0) or 0)
        early_stop_min_delta = float(getattr(self.config.trainer, "early_stop_min_delta", 0.0) or 0.0)
        early_stop_metric = str(getattr(self.config.trainer, "early_stop_metric", "val/accuracy"))
        early_stop_mode = str(getattr(self.config.trainer, "early_stop_mode", "max")).lower()
        if early_stop_mode not in {"min", "max"}:
            raise ValueError(f"trainer.early_stop_mode must be 'min' or 'max', got {early_stop_mode!r}")
        best_val_metric = None
        bad_val_steps = 0

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        start_epoch = global_step // self.steps_per_epoch

        # Each dataloader item is one sample; the collator converts it to
        # 2 * max_pairs_per_sample sequences. Pass sequence counts to engine
        # so gradient-accumulation micro-batch sizes stay consistent.
        micro_bsz = self.config.data.micro_batch_size_per_gpu
        assert micro_bsz % 2 == 0, (
            f"micro_batch_size_per_gpu must be even for RM training (got {micro_bsz}). "
            "Each micro-batch must contain whole pairs."
        )

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            # micro_batch_size_per_gpu is in *sequences*
            "micro_batch_size_per_gpu": micro_bsz,
            "temperature": 1.0,
            # global_batch_size in *sequences*
            "global_batch_size": self.global_batch_size * 2 * self.max_pairs_per_sample,
            "pad_mode": DatasetPadMode.NO_PADDING,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for _step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)
                batch_seqlens_ntd = NonTensorData(batch_seqlens)
                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                if global_step == self.start_profile_step:
                    self.training_client.start_profile()

                output = self.training_client.train_batch(data=data)

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = tu.get(output, "metrics")

                    metrics["train/loss"] = metrics.pop("loss")
                    metrics["train/grad_norm"] = metrics.pop("grad_norm")
                    metrics["train/lr"] = metrics.pop("lr")
                    metrics["train/mfu"] = metrics.pop("mfu")
                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    # Forward RM-specific metrics if present
                    for key in list(metrics.keys()):
                        if key.startswith("rm/"):
                            val = metrics.pop(key)
                            with torch.no_grad():
                                if isinstance(val, (list, tuple, torch.Tensor)):
                                    val = torch.as_tensor(val, dtype=torch.float32).flatten()
                                    val = val.mean().item()
                            metrics[f"train/{key[3:]}"] = val
                    
                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0
                should_stop = False

                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    val_losses = []
                    val_accuracies = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)

                        if self.engine.is_mp_src_rank_with_outputs():
                            val_metrics = tu.get(output, "metrics")
                            val_losses.append(val_metrics["loss"])
                            if "rm/accuracy" in val_metrics:
                                val_accuracies.append(val_metrics["rm/accuracy"])

                    if self.engine.is_mp_src_rank_with_outputs():
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                        torch.distributed.all_reduce(
                            val_loss,
                            op=torch.distributed.ReduceOp.AVG,
                            group=self.engine.get_data_parallel_group(),
                        )
                        val_loss_value = val_loss.detach().item()
                        if len(val_accuracies) > 0:
                            val_acc = torch.mean(torch.tensor(val_accuracies, device=self.device_name))
                            torch.distributed.all_reduce(
                                val_acc,
                                op=torch.distributed.ReduceOp.AVG,
                                group=self.engine.get_data_parallel_group(),
                            )
                            val_accuracy_value = val_acc.detach().item()
                        else:
                            val_accuracy_value = None

                    if is_logging:
                        metric = {"val/loss": val_loss_value}
                        if val_accuracy_value is not None:
                            metric["val/accuracy"] = val_accuracy_value
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric

                    if self.engine.is_mp_src_rank_with_outputs() and early_stop_patience > 0:
                        if early_stop_metric == "val/accuracy":
                            current_metric_value = val_accuracy_value
                        elif early_stop_metric == "val/loss":
                            current_metric_value = val_loss_value
                        else:
                            raise ValueError(f"Unsupported early_stop_metric={early_stop_metric!r}. Falling back to val/loss.")

                        if (
                            best_val_metric is None
                            or (
                                early_stop_mode == "min"
                                and current_metric_value < best_val_metric - early_stop_min_delta
                            )
                            or (
                                early_stop_mode == "max"
                                and current_metric_value > best_val_metric + early_stop_min_delta
                            )
                        ):
                            best_val_metric = current_metric_value
                            bad_val_steps = 0
                        else:
                            bad_val_steps += 1
                        should_stop = bad_val_steps >= early_stop_patience

                    if early_stop_patience > 0:
                        stop_signal = torch.tensor(int(should_stop), device=self.device_name)
                        torch.distributed.all_reduce(stop_signal, op=torch.distributed.ReduceOp.MAX)
                        should_stop = stop_signal.item() > 0

                    torch.distributed.barrier()

                if should_stop:
                    if is_logging:
                        print(
                            f"Early stopping at step {global_step} "
                            f"(best {early_stop_metric}: {best_val_metric:.6f})."
                        )
                    return

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_rm(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = RewardTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="motion_reward_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_rm(config)


if __name__ == "__main__":
    main()
