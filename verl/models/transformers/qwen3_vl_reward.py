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
Qwen3-VL reward model: subclasses Qwen3VLForConditionalGeneration and adds a
scalar reward head.

Design:
  - Inherits all pretrained weights from Qwen3VLForConditionalGeneration.
  - Adds ``score = nn.Linear(hidden_size, num_labels, bias=False)``.
  - ``forward`` calls the parent forward (monkey-patched by verl for packed
    sequences) with ``output_hidden_states=True``, then applies ``score`` to
    the last hidden state.  Returns ``ModelOutput(logits=...)`` where
    ``logits`` has shape ``(..., num_labels)``.
  - Registered with ``AutoModelForTokenClassification`` so
    ``RewardModelWorker._build_model`` can load it via
    ``AutoModelForTokenClassification.from_pretrained``.

Usage (direct):
    model = Qwen3VLRewardModel.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    # or via auto class:
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", num_labels=1
    )
"""

import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Qwen3VLRewardModel(Qwen3VLForConditionalGeneration):
    """Qwen3-VL with a scalar reward head for reward model training.

    Subclasses ``Qwen3VLForConditionalGeneration`` so that:
      - Standard ``from_pretrained`` weight loading works out of the box
        (all backbone weights load; ``score`` is newly initialised).
      - ``apply_monkey_patch`` targets this model directly — it checks
        ``model.config.model_type == "qwen3_vl"`` and patches
        ``Qwen3VLForConditionalGeneration.forward`` at the class level, so
        ``super().forward(...)`` calls the patched (packed-sequence-aware)
        implementation.
      - ``AutoModelForTokenClassification.from_pretrained`` can instantiate
        this class via the registration below.

    The ``lm_head`` from the parent class is present but unused during
    reward scoring.
    """

    config_class = Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        num_labels = getattr(config, "num_labels", 1)
        hidden_size = config.text_config.hidden_size
        self.score = nn.Linear(hidden_size, num_labels, bias=False)
        nn.init.normal_(self.score.weight, std=0.02)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        use_cache: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """Forward pass returning per-token reward scores via ``.logits``.

        Calls the parent forward (which may be monkey-patched by verl for
        packed-sequence / remove-padding mode) with
        ``output_hidden_states=True``, then projects the last hidden state
        through the reward head and maps scores into ``[-1, 1]`` using
        ``2 * sigmoid(x) - 1``.

        Args:
            input_ids:           (bsz, seq_len) or (1, total_nnz) for packed.
            attention_mask:      optional; pass None for packed sequences.
            position_ids:        (4, 1, total_nnz) for packed Qwen3-VL mrope.
            pixel_values:        image pixel values.
            pixel_values_videos: video pixel values.
            image_grid_thw:      image grid shapes.
            video_grid_thw:      video grid shapes.
            use_cache:           always forced to False.

        Returns:
            ModelOutput with ``.logits`` of shape ``(..., num_labels)``.
        """
        kwargs["output_hidden_states"] = True
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=False,
            # output_hidden_states=True,
            **kwargs,
        )
        # hidden_states[-1]: last transformer layer output
        last_hidden = outputs.hidden_states[-1]  # (..., hidden_size)
        raw_scores = self.score(last_hidden)  # (..., num_labels)
        # logits = 2.0 * torch.sigmoid(raw_scores) - 1.0  # (..., num_labels), bounded in [-1, 1]

        return ModelOutput(logits=raw_scores)


# Register so that AutoModelForTokenClassification.from_pretrained("Qwen/Qwen3-VL-*")
# instantiates Qwen3VLRewardModel instead of a generic token-classification wrapper.
# This import side-effect is intentional: any worker that needs to load a Qwen3-VL
# reward model should import this module first.
AutoModelForTokenClassification.register(Qwen3VLConfig, Qwen3VLRewardModel)
