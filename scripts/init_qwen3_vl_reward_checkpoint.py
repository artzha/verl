#!/usr/bin/env python3
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
Initialize a Qwen3-VL reward-model checkpoint from a base Qwen3-VL model.

This script loads base Qwen3-VL weights into `Qwen3VLRewardModel` and saves a
Hugging Face checkpoint directory that can be used by
`reward_model.model.path` in PPO/GRPO.

The reward head (`score`) is randomly initialized by model construction.

Example:
    python external/verl/scripts/init_qwen3_vl_reward_checkpoint.py \
      --base_model Qwen/Qwen3-VL-2B-Instruct \
      --output_dir /tmp/qwen3vl2b_reward_init
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from verl.models.transformers.qwen3_vl_reward import Qwen3VLRewardModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize Qwen3-VL reward-model checkpoint.")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base Qwen3-VL model path or HF repo id (e.g. Qwen/Qwen3-VL-2B-Instruct).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save initialized reward-model checkpoint.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading config/model/tokenizer/processor.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype for loading/saving.",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=1,
        help="Reward head output size (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "float16": "float16",
        "bfloat16": "bfloat16",
        "float32": "float32",
    }
    torch_dtype = dtype_map[args.torch_dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    config.num_labels = args.num_labels
    config.classifier_dropout = 0.0

    model = Qwen3VLRewardModel.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=getattr(torch, torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(args.output_dir)

    try:
        processor = AutoProcessor.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
        )
        processor.save_pretrained(args.output_dir)
    except Exception as exc:
        print(f"[Warning] Failed to save processor: {exc}")

    print(f"Saved initialized reward checkpoint to: {args.output_dir}")
    print("Use it with:")
    print(f"  reward_model.model.path={args.output_dir}")
    print("  reward_model.model.external_lib=verl.models.transformers.qwen3_vl_reward")


if __name__ == "__main__":
    main()

