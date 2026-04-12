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
import logging
import os
import copy
import argparse
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.models.transformers.qwen3_vl import get_rope_index
from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.dataset.multiturn_sft_dataset import (convert_nested_value_to_list_recursive, print_assembled_message)


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class RMDataset(Dataset):
    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: Optional[DictConfig] = None,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        config = config or {}
        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self.prompt_key = config.get("prompt_key", "prompt")
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")
        self.image_key = config.get("image_key", "image")
        self.video_key = config.get("video_key", "video")

        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        assert self.truncation in ["error", "left", "right"]
        self.max_length = config.get("max_length", 1024)
        self.return_messages = config.get("return_messages", False)

        self.max_samples = max_samples
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed", None)
        self.image_patch_size = config.get(
            "image_patch_size",
            processor.image_processor.patch_size if processor is not None else None,
        )
        self.resize_mode = config.get("resize_mode", "auto")
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)
        self.apply_chat_template_kwargs = {}
        for k, v in config.get("apply_chat_template_kwargs", {}).items():
            self.apply_chat_template_kwargs[k] = v
            if OmegaConf.is_config(v):
                self.apply_chat_template_kwargs[k] = OmegaConf.to_container(v, resolve=True)

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframes.append(pd.read_parquet(parquet_file))
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        required_cols = {self.prompt_key, self.chosen_key, self.rejected_key}
        missing_cols = required_cols.difference(self.dataframe.columns)
        if missing_cols:
            raise ValueError(f"Missing required RM columns: {sorted(missing_cols)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict[str, Any], branch_key: str) -> list[dict[str, Any]]:
        prompt = convert_nested_value_to_list_recursive(example[self.prompt_key])
        if not isinstance(prompt, list) or len(prompt) == 0:
            raise ValueError(f"{self.prompt_key} must be a non-empty list of messages.")

        branch_message = convert_nested_value_to_list_recursive(example[branch_key])
        if not isinstance(branch_message, dict):
            raise ValueError(f"{branch_key} must be a message dict, got {type(branch_message)}")
        if branch_message.get("role") != "assistant":
            raise ValueError(f"{branch_key} role must be 'assistant', got {branch_message.get('role')!r}")
        if not isinstance(branch_message.get("content"), str):
            raise ValueError(f"{branch_key} content must be str, got {type(branch_message.get('content'))}")

        messages = copy.deepcopy(prompt)
        messages.append({"role": "assistant", "content": branch_message["content"]})

        images = convert_nested_value_to_list_recursive(example[self.image_key]) if self.image_key in example else []
        videos = convert_nested_value_to_list_recursive(example[self.video_key]) if self.video_key in example else []
        if self.processor is None and (images or videos):
            raise ValueError("processor is required for multimodal RM rows with image/video payloads.")

        image_offset, video_offset = 0, 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            content_list = []
            segments = [item for item in re.split("(<image>|<video>)", content) if item != ""]
            for segment in segments:
                if segment == "<image>":
                    if self.processor is None:
                        raise ValueError("processor is required to process <image> placeholders")
                    image = process_image(
                        images[image_offset],
                        image_patch_size=self.image_patch_size,
                        resize_mode=self.resize_mode,
                    )
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    if self.processor is None:
                        raise ValueError("processor is required to process <video> placeholders")
                    videos_item = videos[video_offset]
                    if isinstance(videos_item, np.ndarray):
                        videos_item = videos_item.tolist()
                    if not isinstance(videos_item, dict) or "video" not in videos_item:
                        raise ValueError(f"video input must be dict containing 'video', got {type(videos_item)}")
                    videos_item = dict(videos_item)
                    if isinstance(videos_item["video"], np.ndarray):
                        videos_item["video"] = videos_item["video"].tolist()
                    video = process_video(
                        videos_item,
                        image_patch_size=self.image_patch_size,
                        resize_mode=self.resize_mode,
                    )
                    content_list.append({"type": "video", "video": video})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        if image_offset != len(images):
            raise ValueError(f"image placeholders mismatch: used {image_offset}, provided {len(images)}.")
        if video_offset != len(videos):
            raise ValueError(f"video placeholders mismatch: used {video_offset}, provided {len(videos)}.")
        return messages

    def _process_single_message(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if self.processor is not None and self.processor.__class__.__name__ == "Qwen3VLProcessor":
            apply_chat_template_kwargs.setdefault("do_sample_frames", False)

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=inputs.get("image_grid_thw", None),
                video_grid_thw=inputs.get("video_grid_thw", None),
                second_per_grid_ts=inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_len)
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_len)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        sequence_length = input_ids.shape[0]
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                input_ids = torch.cat((input_ids, padded_input_ids), dim=0)
                attention_mask = torch.cat((attention_mask, padded_attention_mask), dim=0)
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if sequence_length > self.max_length:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")
        return input_ids, attention_mask, position_ids, inputs

    def __getitem__(self, item):
        row_dict: dict[str, Any] = self.dataframe.iloc[item].to_dict()
        tools, enable_thinking = None, None

        chosen_messages = self._build_messages(example=row_dict, branch_key=self.chosen_key)
        rejected_messages = self._build_messages(example=row_dict, branch_key=self.rejected_key)

        chosen_input_ids, chosen_attention_mask, chosen_position_ids, chosen_mm = self._process_single_message(
            chosen_messages
        )
        rejected_input_ids, rejected_attention_mask, rejected_position_ids, rejected_mm = self._process_single_message(
            rejected_messages
        )

        # Debug print once (function is once-decorated in multiturn_sft_dataset.py).
        print_assembled_message(
            self.tokenizer,
            chosen_messages,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_attention_mask,
            tools,
        )

        # Pairwise RM samples must stack chosen/rejected. Under no_padding mode, lengths
        # can differ by one or more tokens (e.g., EOS/chat-template differences), so pad
        # both branches to a shared per-sample length before stacking.
        chosen_len = int(chosen_input_ids.shape[-1])
        rejected_len = int(rejected_input_ids.shape[-1])
        pair_max_len = max(chosen_len, rejected_len)
        if chosen_len != rejected_len:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            if chosen_len < pair_max_len:
                pad = pair_max_len - chosen_len
                chosen_input_ids = torch.cat(
                    (chosen_input_ids, torch.full((pad,), pad_token_id, dtype=chosen_input_ids.dtype)),
                    dim=-1,
                )
                chosen_attention_mask = torch.cat(
                    (chosen_attention_mask, torch.zeros((pad,), dtype=chosen_attention_mask.dtype)),
                    dim=-1,
                )
                chosen_position_ids = F.pad(chosen_position_ids, (0, pad), value=0)
            if rejected_len < pair_max_len:
                pad = pair_max_len - rejected_len
                rejected_input_ids = torch.cat(
                    (rejected_input_ids, torch.full((pad,), pad_token_id, dtype=rejected_input_ids.dtype)),
                    dim=-1,
                )
                rejected_attention_mask = torch.cat(
                    (rejected_attention_mask, torch.zeros((pad,), dtype=rejected_attention_mask.dtype)),
                    dim=-1,
                )
                rejected_position_ids = F.pad(rejected_position_ids, (0, pad), value=0)

        result = {
            "input_ids": torch.stack((chosen_input_ids, rejected_input_ids), dim=0),
            "attention_mask": torch.stack((chosen_attention_mask, rejected_attention_mask), dim=0),
            "position_ids": torch.stack((chosen_position_ids, rejected_position_ids), dim=0),
        }
        if self.return_messages:
            result["messages"] = {"chosen": chosen_messages, "rejected": rejected_messages}

        # Store chosen and rejected multi-modal inputs separately so that the reward
        # trainer collator can interleave them correctly (chosen_0, rejected_0, ...).
        chosen_multi_modal_inputs: dict[str, torch.Tensor] = {}
        rejected_multi_modal_inputs: dict[str, torch.Tensor] = {}
        for key in set(chosen_mm.keys()).union(set(rejected_mm.keys())):
            chosen_val = chosen_mm.get(key, None)
            rejected_val = rejected_mm.get(key, None)
            if chosen_val is None or rejected_val is None:
                continue
            if not isinstance(chosen_val, torch.Tensor) or not isinstance(rejected_val, torch.Tensor):
                continue
            if chosen_val.shape[1:] != rejected_val.shape[1:]:
                # Match MultiTurnSFTDataset behavior: drop inconsistent tensor keys.
                continue
            chosen_multi_modal_inputs[key] = chosen_val
            rejected_multi_modal_inputs[key] = rejected_val

        if len(chosen_multi_modal_inputs) > 0:
            result["chosen_multi_modal_inputs"] = chosen_multi_modal_inputs
            result["rejected_multi_modal_inputs"] = rejected_multi_modal_inputs
        return result
    
    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):
        """Check concatenated input_ids of apply_chat_template to each turn equals
        apply_chat_template to whole messages.
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        if self.processor is not None and self.processor.__class__.__name__ == "Qwen3VLProcessor":
            apply_chat_template_kwargs.setdefault("do_sample_frames", False)
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                logger.warning_once(error_message)
            else:
                raise AssertionError(error_message)


if __name__ == "__main__":
    from pathlib import Path

    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer

    def parse_args():
        ap = argparse.ArgumentParser(description="Smoke test for multimodal RMDataset.")
        ap.add_argument(
            "--config",
            type=str,
            default="external/verl/verl/trainer/config/motion_reward_trainer.yaml",
            help="YAML config path (reads `data` section).",
        )
        ap.add_argument(
            "--parquet",
            action="append",
            default=[],
            help="Override parquet path(s). Repeat for multiple files.",
        )
        ap.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="Tokenizer/processor model path override.",
        )
        ap.add_argument(
            "--max-samples",
            type=int,
            default=4,
            help="Maximum dataset samples for the smoke test.",
        )
        return ap.parse_args()

    args = parse_args()
    cfg = OmegaConf.load(args.config) if Path(args.config).exists() else OmegaConf.create({})
    OmegaConf.resolve(cfg)
    cfg.data.train_files = "data/unified_motion_rm_v1/parquet/train/train.parquet"
    cfg.data.val_files = "data/unified_motion_rm_v1/parquet/val/val.parquet"
    data_cfg = cfg.get("data", {})
    

    if args.parquet:
        parquet_files = args.parquet
    else:
        train_files = data_cfg.get("train_files", "")
        if isinstance(train_files, str) and train_files:
            parquet_files = [train_files]
        elif isinstance(train_files, (list, ListConfig)):
            parquet_files = list(train_files)
        else:
            raise ValueError("No parquet files provided. Use --parquet or set data.train_files in config.")

    model_path = args.model_path
    if model_path is None:
        model_cfg = cfg.get("model", {})
        model_path = model_cfg.get("tokenizer_path") or model_cfg.get("path")
    if model_path is None:
        raise ValueError("Missing model path. Provide --model-path or set model.path/tokenizer_path in config.")

    trust_remote_code = bool(cfg.get("model", {}).get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    dataset = RMDataset(
        parquet_files=parquet_files,
        tokenizer=tokenizer,
        processor=processor,
        config=data_cfg,
        max_samples=args.max_samples,
    )
    print(f"dataset len: {len(dataset)}")
    sample = dataset[0]
    print("sample keys:", list(sample.keys()))
    print("input_ids shape:", tuple(sample["input_ids"].shape))
    print("attention_mask shape:", tuple(sample["attention_mask"].shape))
    print("position_ids shape:", tuple(sample["position_ids"].shape))
    for mm_key in ("chosen_multi_modal_inputs", "rejected_multi_modal_inputs"):
        if mm_key in sample:
            print(f"{mm_key} keys:", list(sample[mm_key].keys()))
            for k, v in sample[mm_key].items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}")
